#include "falcon_common.h"
#include "libfalcon.h"
#include "build-info.h"

#include <cmath>
#include <ctime>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size());
    float max_logit = logits[0];
    for (float v : logits) max_logit = std::max(max_logit, v);
    double sum_exp = 0.0;
    for (size_t i = 0; i < logits.size(); i++) {
        // Subtract the maximum logit value from the current logit value for numerical stability
        const float logit = logits[i] - max_logit;
        const float exp_logit = expf(logit);
        sum_exp += exp_logit;
        probs[i] = exp_logit;
    }
    for (size_t i = 0; i < probs.size(); i++) probs[i] /= sum_exp;
    return probs;
}

void perplexity(falcon_context * ctx, const gpt_params & params) {
    // Download: https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip?ref=salesforce-research
    // Run `./perplexity -m models/7B/ggml-model-q4_0.bin -f wiki.test.raw`
    // Output: `perplexity: 13.5106 [114/114]`
    // BOS tokens will be added for each chunk before eval
    auto tokens = ::falcon_tokenize(ctx, params.prompt, true);

    int count   = 0;

    const int n_chunk = tokens.size() / params.n_ctx;
    const int n_vocab = falcon_n_vocab(ctx);
    const int n_batch = params.n_batch;

    double nll = 0.0;
    fprintf(stderr, "%s: calculating perplexity over %d chunks, batch_size=%d\n", __func__, n_chunk, n_batch);

    for (int i = 0; i < n_chunk; ++i) {
        const int start =     i * params.n_ctx;
        const int end   = start + params.n_ctx;

        const int num_batches = (params.n_ctx + n_batch - 1) / n_batch;

        std::vector<float> logits;

        const auto t_start = std::chrono::high_resolution_clock::now();

        for (int j = 0; j < num_batches; ++j) {
            const int batch_start = start + j * n_batch;
            const int batch_size  = std::min(end - batch_start, n_batch);

            // save original token and restore it after eval
            const auto token_org = tokens[batch_start];

            // add BOS token for the first batch of each chunk
            if (j == 0) {
                // tokens[batch_start] = falcon_token_bos();
            }
            falcon_evaluation_config configuration;
                configuration.n_past = j * n_batch;
                configuration.n_tokens = batch_size;
                configuration.debug_timings = params.debug_timings;
                configuration.n_threads = params.n_threads;
            if (falcon_eval(ctx, tokens.data() + batch_start, configuration)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return;
            }

            // restore the original token in case it was set to BOS
            tokens[batch_start] = token_org;

            const auto batch_logits = falcon_get_logits(ctx);
            logits.insert(logits.end(), batch_logits, batch_logits + batch_size * n_vocab);
        }

        const auto t_end = std::chrono::high_resolution_clock::now();

        if (i == 0) {
            const float t_total = std::chrono::duration<float>(t_end - t_start).count();
            fprintf(stderr, "%s: %.2f seconds per pass - ETA ", __func__, t_total);
            int total_seconds = (int)(t_total * n_chunk);
            if (total_seconds >= 60*60) {
                fprintf(stderr, "%d hours ", total_seconds / (60*60));
                total_seconds = total_seconds % (60*60);
            }
            fprintf(stderr, "%d minutes\n", total_seconds / 60);
        }

        // We get the logits for all the tokens in the context window (params.n_ctx)
        // from llama_eval above.  Now, based on https://huggingface.co/docs/transformers/perplexity,
        // calculate the perplexity over the last half of the window (so the model always has
        // some context to predict the token).
        //
        // We rely on the fact that attention in the forward pass only looks at previous
        // tokens here, so the logits returned for each token are an accurate representation
        // of what the model would have predicted at that point.
        //
        // Example, we have a context window of 512, we will compute perplexity for each of the
        // last 256 tokens.  Then, we split the input up into context window size chunks to
        // process the entire prompt.
        for (int j = std::min(512, params.n_ctx / 2); j < params.n_ctx - 1; ++j) {
            // Calculate probability of next token, given the previous ones.
            const std::vector<float> tok_logits(
                logits.begin() + (j + 0) * n_vocab,
                logits.begin() + (j + 1) * n_vocab);

            const float prob = softmax(tok_logits)[tokens[start + j + 1]];

            nll += -std::log(prob);
            ++count;
        }
        // perplexity is e^(average negative log-likelihood)
        printf("[%d]%.4lf,", i + 1, std::exp(nll / count));
        fflush(stdout);
    }
    printf("\n");
}

int main(int argc, char ** argv) {
    gpt_params params;

    params.n_batch = 512;
    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    params.perplexity = true;
    params.n_batch = std::min(params.n_batch, params.n_ctx);

    if (params.n_ctx > 2048) {
        fprintf(stderr, "%s: warning: model does not support context sizes greater than 2048 tokens (%d specified);"
                "expect poor results\n", __func__, params.n_ctx);
    }

    fprintf(stderr, "%s: build = %d (%s)\n", __func__, BUILD_NUMBER, BUILD_COMMIT);

    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed  = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    falcon_init_backend();

    falcon_context * ctx;

    // load the model and apply lora adapter, if any
    ctx = falcon_init_from_gpt_params(params);
    if (ctx == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    #if defined(GGML_USE_CUBLAS)
    // wait for cublas and show device information
    {
        ggml_cuda_print_gpu_status(ggml_cuda_get_system_gpu_status(),true);
    }
    #endif

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s\n",
                 falcon_print_system_info(params.n_threads, std::thread::hardware_concurrency()));
    }

    perplexity(ctx, params);

    falcon_print_timings(ctx);
    llama_free(ctx);

    return 0;
}
