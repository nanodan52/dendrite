#include "llama.h"
#include <clocale>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [-o output.bin] [-s system_prompt] [--no-think] [prompt]\n", argv[0]);
    printf("\n");
    printf("Generates text and captures per-token hidden states to a binary file.\n");
    printf("Output: raw float32 vectors, shape [n_tokens, n_embd]\n");
    printf("Metadata written to <output>.json\n");
    printf("\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string prompt = "Hello my name is";
    std::string output_path = "hidden_states.bin";
	std::string system_prompt;
    int ngl = 99;
    int n_predict = 1280;
	bool no_think = false;

    // parse command line arguments
    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) { model_path = argv[++i]; }
                else { print_usage(argc, argv); return 1; }
            } else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) {
                    try { n_predict = std::stoi(argv[++i]); }
                    catch (...) { print_usage(argc, argv); return 1; }
                } else { print_usage(argc, argv); return 1; }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    try { ngl = std::stoi(argv[++i]); }
                    catch (...) { print_usage(argc, argv); return 1; }
                } else { print_usage(argc, argv); return 1; }
            } else if (strcmp(argv[i], "-o") == 0) {
                if (i + 1 < argc) { output_path = argv[++i]; }
                else { print_usage(argc, argv); return 1; }
			} else if (strcmp(argv[i], "-s") == 0) {
                if (i + 1 < argc) { system_prompt = argv[++i]; }
                else { print_usage(argc, argv); return 1; }
            } else if (strcmp(argv[i], "--no-think") == 0) {
				no_think = true;
			} else if (strcmp(argv[i], "-f") == 0) {
                if (i + 1 < argc) {
                    std::ifstream file(argv[++i]);
                    if (!file.is_open()) {
                        fprintf(stderr, "Error: cannot open prompt file '%s'\n", argv[i]);
                        return 1;
                    }
                    prompt = std::string((std::istreambuf_iterator<char>(file)),
                                          std::istreambuf_iterator<char>());
                } else { print_usage(argc, argv); return 1; }
			} else if (strcmp(argv[i], "--sf") == 0) {
                if (i + 1 < argc) {
                    std::ifstream file(argv[++i]);
                    if (!file.is_open()) {
                        fprintf(stderr, "Error: cannot open system prompt file '%s'\n", argv[i]);
                        return 1;
                    }
                    system_prompt = std::string((std::istreambuf_iterator<char>(file)),
                                                 std::istreambuf_iterator<char>());
                } else { print_usage(argc, argv); return 1; }
            } else {
                break;
            }
        }
        if (model_path.empty()) { print_usage(argc, argv); return 1; }
        if (i < argc) {
            prompt = argv[i++];
            for (; i < argc; i++) { prompt += " "; prompt += argv[i]; }
        }
    }

    // load backends
    ggml_backend_load_all();

    // load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd(model);

    // build prompt with chat template (always no-think)
    std::string formatted_prompt;
    if (!system_prompt.empty()) {
        formatted_prompt = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
    }
    formatted_prompt += "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    if (!no_think) {
        formatted_prompt += "<think>\n";
    }

    const int n_prompt = -llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.size(), NULL, 0, false, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.size(), prompt_tokens.data(), prompt_tokens.size(), false, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }
    // create context with hidden_states enabled
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_prompt + n_predict;
    ctx_params.n_batch = n_prompt;
    ctx_params.no_perf = false;
    ctx_params.hidden_states = true;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
        return 1;
    }

    // sampler — greedy for reproducibility
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    if (no_think) {
        llama_logit_bias biases[2] = {
            { 248068, -INFINITY },
            { 248066, -INFINITY },
        };
        llama_sampler_chain_add(smpl, llama_sampler_init_logit_bias(
            llama_vocab_n_tokens(vocab), 2, biases));
    }

    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // open output file
    FILE * f_out = fopen(output_path.c_str(), "wb");
    if (!f_out) {
        fprintf(stderr, "%s: error: failed to open output file '%s'\n", __func__, output_path.c_str());
        return 1;
    }

    // storage for generated token info (for metadata)
    std::vector<llama_token> generated_tokens;
    std::vector<std::string> generated_pieces;
    int n_captured = 0;

    // print prompt
    fprintf(stderr, "prompt: '%s'\n", prompt.c_str());
    fprintf(stderr, "n_embd: %d\n", n_embd);
    fprintf(stderr, "generating %d tokens...\n\n", n_predict);

    for (auto id : prompt_tokens) {
        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n > 0) { printf("%.*s", n, buf); }
    }

    // eval prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s: error: failed to eval prompt\n", __func__);
        fclose(f_out);
        return 1;
    }

    // capture hidden state after prompt eval (the "launch point")
    {
        const float * embd = llama_get_embeddings_ith(ctx, -1);
        if (embd) {
            fwrite(embd, sizeof(float), n_embd, f_out);
            n_captured++;
        } else {
            fprintf(stderr, "%s: warning: failed to get embeddings after prompt eval\n", __func__);
        }
    }

    // sample first token
    llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);

    // generation loop
    int n_decode = 0;
    while (!llama_vocab_is_eog(vocab, new_token_id) && n_decode < n_predict) {
        // record token
        generated_tokens.push_back(new_token_id);
        char buf[128];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            generated_pieces.push_back(std::string(buf, n));
            printf("%.*s", n, buf);
            fflush(stdout);
        } else {
            generated_pieces.push_back("");
        }

        // decode the new token
        batch = llama_batch_get_one(&new_token_id, 1);
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "\n%s: error: failed to eval token at position %d\n", __func__, n_decode);
            break;
        }

        // capture hidden state
        {
            const float * embd = llama_get_embeddings_ith(ctx, -1);
            if (embd) {
                fwrite(embd, sizeof(float), n_embd, f_out);
                n_captured++;
            } else {
                fprintf(stderr, "\n%s: warning: failed to get embeddings at step %d\n", __func__, n_decode);
            }
        }

        // sample next token
        new_token_id = llama_sampler_sample(smpl, ctx, -1);
        n_decode++;
    }

    printf("\n");
    fclose(f_out);

    // write metadata
    std::string meta_path = output_path + ".json";
    FILE * f_meta = fopen(meta_path.c_str(), "w");
    if (f_meta) {
        fprintf(f_meta, "{\n");
        fprintf(f_meta, "  \"model\": \"%s\",\n", model_path.c_str());
        fprintf(f_meta, "  \"n_embd\": %d,\n", n_embd);
        fprintf(f_meta, "  \"n_captured\": %d,\n", n_captured);
        fprintf(f_meta, "  \"prompt\": \"");
        // escape the prompt for JSON
        for (char c : prompt) {
            if (c == '"') fprintf(f_meta, "\\\"");
            else if (c == '\\') fprintf(f_meta, "\\\\");
            else if (c == '\n') fprintf(f_meta, "\\n");
            else fprintf(f_meta, "%c", c);
        }
        fprintf(f_meta, "\",\n");
        fprintf(f_meta, "  \"tokens\": [\n");
        for (size_t i = 0; i < generated_tokens.size(); i++) {
            fprintf(f_meta, "    {\"id\": %d, \"piece\": \"", generated_tokens[i]);
            for (char c : generated_pieces[i]) {
                if (c == '"') fprintf(f_meta, "\\\"");
                else if (c == '\\') fprintf(f_meta, "\\\\");
                else if (c == '\n') fprintf(f_meta, "\\n");
                else fprintf(f_meta, "%c", c);
            }
            fprintf(f_meta, "\"}%s\n", i + 1 < generated_tokens.size() ? "," : "");
        }
        fprintf(f_meta, "  ]\n");
        fprintf(f_meta, "}\n");
        fclose(f_meta);
        fprintf(stderr, "\nmetadata written to %s\n", meta_path.c_str());
    }

    fprintf(stderr, "captured %d hidden state vectors of dimension %d\n", n_captured, n_embd);
    fprintf(stderr, "output written to %s (%.2f MB)\n", output_path.c_str(),
            (n_captured * n_embd * sizeof(float)) / (1024.0 * 1024.0));

    llama_perf_context_print(ctx);

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}