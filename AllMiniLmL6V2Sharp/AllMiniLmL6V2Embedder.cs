using AllMiniLmL6V2Sharp.Tokenizer;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AllMiniLmL6V2Sharp
{

    /// <summary>
    /// Generate Embeddings via All-MiniLM-L6-v2
    /// </summary>
    public class AllMiniLmL6V2Embedder : IEmbedder
    {
        private readonly ITokenizer _tokenizer;
        private readonly string _modelPath;
        private readonly bool _truncate;
        /// <summary>
        /// Initializes the AllMiniLmL6v2 Embedder
        /// </summary>
        /// <param name="modelPath">Path to the embedding onnx model.</param>
        /// <param name="tokenizer">Optional custom tokenizer function.</param>
        /// <param name="truncate">If true, automatically truncates tokens to 512 tokens.</param>
        public AllMiniLmL6V2Embedder(string modelPath = "./model/model.onnx", ITokenizer? tokenizer = null, bool truncate = false)
        {
            _tokenizer = tokenizer ?? new BertTokenizer("./model/vocab.txt");
            _modelPath = modelPath;
            _truncate = truncate;
        }

        /// <summary>
        /// Generates an embedding array for the given sentance.
        /// </summary>
        /// <param name="sentence">Text to embed.</param>
        /// <returns>Sentance embeddings</returns>
        public IEnumerable<float> GenerateEmbedding(string sentence)
        {
            using RunOptions runOptions = new RunOptions();
            using InferenceSession session = new InferenceSession(_modelPath);

            return GenerateEmbedding(sentence, runOptions, session);
        }
        private IEnumerable<float> GenerateEmbedding(string sentence, RunOptions runOptions, InferenceSession session)
        {
            // Tokenize Input
            IEnumerable<Token> tokens = _tokenizer.Tokenize(sentence);
            if (_truncate && tokens.Count() > 512)
            {
                tokens = tokens.Take(512);
            }

            IEnumerable<EncodedToken> encodedTokens = _tokenizer.Encode(tokens.Count(), sentence);

            // Compute Token Embeddings
            BertInput bertInput = new BertInput
            {
                InputIds = encodedTokens.Select(t => t.InputIds).ToArray(),
                TypeIds = encodedTokens.Select(t => t.TokenTypeIds).ToArray(),
                AttentionMask = encodedTokens.Select(t => t.AttentionMask).ToArray()
            };

            // Create input tensors over the input data.
            using OrtValue inputIdsOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.InputIds,
                  new long[] { 1, bertInput.InputIds.Length });

            using OrtValue attMaskOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.AttentionMask,
                  new long[] { 1, bertInput.AttentionMask.Length });

            using OrtValue typeIdsOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.TypeIds,
                  new long[] { 1, bertInput.TypeIds.Length });

            // Create input data for session. Request all outputs in this case.
            IReadOnlyDictionary<string, OrtValue> inputs = new Dictionary<string, OrtValue>
            {
                { "input_ids", inputIdsOrtValue },
                { "attention_mask", attMaskOrtValue },
                { "token_type_ids", typeIdsOrtValue }
            };

            using IDisposableReadOnlyCollection<OrtValue> output = session.Run(runOptions, inputs, session.OutputNames);

            // Perform Pooling
            var pooled = SingleMeanPooling(output.First(), attMaskOrtValue);

            // Normalize Embeddings
            var normalized = pooled.Normalize(p: 2, dim: 1);

            var result = normalized.ToArray();
            return result;
        }

        /// <summary>
        /// Generates an embedding array for the given sentances.
        /// </summary>
        /// <param name="sentence">Text to embed.</param>
        /// <returns>An enumerable of embeddings.</returns>
        public IEnumerable<IEnumerable<float>> GenerateEmbeddings(IEnumerable<string> sentences)
        {
            using RunOptions runOptions = new RunOptions();
            using InferenceSession session = new InferenceSession(_modelPath);

            foreach (var sentence in sentences)
            {
                yield return GenerateEmbedding(sentence, runOptions, session);
            }
        }

        public async IAsyncEnumerable<(string sentence, float[] embedding)> GenerateEmbeddingsAsync(IEnumerable<string> sentences)
        {
            using RunOptions runOptions = new RunOptions();
            using InferenceSession session = new InferenceSession(_modelPath);

            await Task.Yield();

            var tasks = sentences.AsParallel()
                .WithExecutionMode(ParallelExecutionMode.ForceParallelism)
                .Select(async sentence =>
            {
                await Task.Yield();
                var embedding = GenerateEmbedding(sentence, runOptions, session).ToArray();
                return (sentence, embedding);
            });

            foreach (var task in tasks)
            {
                yield return await task;
            }
        }

        private DenseTensor<float> SingleMeanPooling(OrtValue modelOutput, OrtValue attentionMask)
        {
            DenseTensor<float> tokenTensor = OrtToTensor<float>(modelOutput);
            DenseTensor<float> maskTensor = AttentionMaskToTensor(attentionMask);
            return MeanPooling(tokenTensor, maskTensor);
        }

        private static DenseTensor<float> AttentionMaskToTensor(OrtValue attentionMask)
        {
            DenseTensor<long> maskIntTensor = OrtToTensor<long>(attentionMask);
            var maskFloatData = maskIntTensor.Select(x => (float)x).ToArray();
            DenseTensor<float> maskTensor = new DenseTensor<float>(maskFloatData, maskIntTensor.Dimensions);
            return maskTensor;
        }

        private DenseTensor<float> MeanPooling(DenseTensor<float> tokenTensor, DenseTensor<float> maskTensor)
        {
            DenseTensor<float> maskedSum = ApplyMaskAndSum(tokenTensor, maskTensor);
            return maskedSum;
        }

        private DenseTensor<float> ApplyMaskAndSum(DenseTensor<float> tokenTensor, DenseTensor<float> maskTensor)
        {
            var expanded = maskTensor.Unsqueeze(-1).Expand(tokenTensor.Dimensions.ToArray());

            var multiplied = tokenTensor.ElementWiseMultiply(expanded);

            var sum = multiplied.Sum(1);

            var sumMask = expanded.Sum(1);

            var clampedMask = sumMask.Clamp(min: 1e-9f);

            var result = sum.ElementWiseDivide(clampedMask);

            return result;
        }

        private static DenseTensor<T> OrtToTensor<T>(OrtValue value) where T : unmanaged
        {
            var typeAndShape = value.GetTensorTypeAndShape();
            var tokenShape = new ReadOnlySpan<int>(typeAndShape.Shape.Select(s => (int)s).ToArray());
            var tokenEmbeddings = value.GetTensorDataAsSpan<T>();
            DenseTensor<T> tokenTensor = new DenseTensor<T>(tokenShape);
            tokenEmbeddings.CopyTo(tokenTensor.Buffer.Span);
            return tokenTensor;
        }
    }
}
