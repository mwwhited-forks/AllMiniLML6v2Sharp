using AllMiniLmL6V2Sharp.Tokenizer;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

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
        /// Generates an embedding array for the given sentence.
        /// </summary>
        /// <param name="sentence">Text to embed.</param>
        /// <returns>Sentence embeddings</returns>
        public float[] GenerateEmbedding(string sentence)
        {
            // Tokenize Input
            var tokens = _tokenizer.Tokenize(sentence);
            if (_truncate && tokens.Length > 512)
            {
                tokens = tokens.Take(512).ToArray();
            }

            var encodedTokens = _tokenizer.Encode(tokens.Length, sentence);

            // Compute Token Embeddings
            var bertInput = new BertInput
            {
                InputIds = encodedTokens.Select(t => t.InputIds).ToArray(),
                TypeIds = encodedTokens.Select(t => t.TokenTypeIds).ToArray(),
                AttentionMask = encodedTokens.Select(t => t.AttentionMask).ToArray(),
            };

            using var runOptions = new RunOptions();
            using var session = new InferenceSession(_modelPath);

            // Create input tensors over the input data.
            using var inputIdsOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.InputIds,
                  new long[] { 1, bertInput.InputIds.Length });

            using var attMaskOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.AttentionMask,
                  new long[] { 1, bertInput.AttentionMask.Length });

            using var typeIdsOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.TypeIds,
                  new long[] { 1, bertInput.TypeIds.Length });

            // Create input data for session. Request all outputs in this case.
            var inputs = new Dictionary<string, OrtValue>
            {
                { "input_ids", inputIdsOrtValue },
                { "attention_mask", attMaskOrtValue },
                { "token_type_ids", typeIdsOrtValue }
            };

            using var output = session.Run(runOptions, inputs, session.OutputNames);

            // Perform Pooling
            var pooled = SingleMeanPooling(output.First(), attMaskOrtValue);

            // Normalize Embeddings
            var normalized = pooled.Normalize(p: 2, dim: 1);

            var result = normalized.ToArray();
            return result;
        }

        /// <summary>
        /// Generates an embedding array for the given sentences.
        /// </summary>
        /// <param name="sentence">Text to embed.</param>
        /// <returns>An enumerable of embeddings.</returns>
        public IEnumerable<(string sentence, float[] embedding)> GenerateEmbeddings(IEnumerable<string> sentences)
        {
            // Tokenize Input
            var allTokens = new List<IEnumerable<Token>>();
            var allEncoded = new List<IEnumerable<EncodedToken>>();

            foreach (var sentence in sentences)
            {
                var tokens = _tokenizer.Tokenize(sentence);

                if (_truncate && tokens.Length > 512)
                {
                    tokens = tokens.Take(512).ToArray();
                }

                allTokens.Add(tokens);
            }

            int maxSequence = allTokens.Max(t => t.Count());

            foreach (var sentence in sentences)
            {
                var encodedTokens = _tokenizer.Encode(maxSequence, sentence);
                allEncoded.Add(encodedTokens);
            }

            // Compute Token Embeddings
            var inputs = allEncoded.Select(e => new BertInput
            {
                InputIds = e.Select(t => t.InputIds).ToArray(),
                TypeIds = e.Select(t => t.TokenTypeIds).ToArray(),
                AttentionMask = e.Select(t => t.AttentionMask).ToArray()
            });

            using var runOptions = new RunOptions();
            using var session = new InferenceSession(_modelPath);

            // Create input tensors over the input data.
            var size = inputs.Count();
            var inputIds = inputs.SelectMany(i => i.InputIds).ToArray();
            using var inputIdsOrtValue = OrtValue.CreateTensorValueFromMemory(inputIds,
                  new long[] { size, maxSequence });

            var attentionMask = inputs.SelectMany(i => i.AttentionMask).ToArray();
            using var attMaskOrtValue = OrtValue.CreateTensorValueFromMemory(attentionMask,
                  new long[] { size, inputs.First().AttentionMask.Length });

            var typeIds = inputs.SelectMany(i => i.TypeIds).ToArray();
            using var typeIdsOrtValue = OrtValue.CreateTensorValueFromMemory(typeIds,
                  new long[] { size, maxSequence });

            // Create input data for session. Request all outputs in this case.
            IReadOnlyDictionary<string, OrtValue> ortInputs = new Dictionary<string, OrtValue>
            {
                { "input_ids", inputIdsOrtValue },
                { "attention_mask", attMaskOrtValue },
                { "token_type_ids", typeIdsOrtValue }
            };

            using IDisposableReadOnlyCollection<OrtValue> output = session.Run(runOptions, ortInputs, session.OutputNames);

            // For now, perform this separately for each output value.
            var results = MultiplePostProcess(output.First(), attMaskOrtValue);
            var zipped = sentences.Zip(results, (sentence, result) => (sentence, result));
            return zipped;
        }

        private float[][] MultiplePostProcess(OrtValue modelOutput, OrtValue attentionMask)
        {
            var results = new List<float[]>();
            var output = modelOutput.GetTensorDataAsSpan<float>().ToArray();
            var dimensions = modelOutput.GetTensorTypeAndShape().Shape.Select(s => (int)s).ToArray();
            dimensions[0] = 1;
            long shape = dimensions[0] * dimensions[1] * dimensions[2];

            for (long i = 0; i < output.Length; i += shape)
            {
                var buffer = new float[shape];
                Array.Copy(output, i, buffer, 0, shape);
                var tokenTensor = new DenseTensor<float>(buffer, dimensions);
                var maskTensor = AttentionMaskToTensor(attentionMask);
                var pooled = MeanPooling(tokenTensor, maskTensor);
                // Normalize Embeddings
                var normalized = pooled.Normalize(p: 2, dim: 1);
                results.Add(normalized.ToArray());
            }

            return results.ToArray();
        }

        private DenseTensor<float> SingleMeanPooling(OrtValue modelOutput, OrtValue attentionMask)
        {
            var tokenTensor = OrtToTensor<float>(modelOutput);
            var maskTensor = AttentionMaskToTensor(attentionMask);
            return MeanPooling(tokenTensor, maskTensor);
        }

        private static DenseTensor<float> AttentionMaskToTensor(OrtValue attentionMask)
        {
            var maskIntTensor = OrtToTensor<long>(attentionMask);
            var maskFloatData = maskIntTensor.Select(x => (float)x).ToArray();
            var maskTensor = new DenseTensor<float>(maskFloatData, maskIntTensor.Dimensions);
            return maskTensor;
        }

        private DenseTensor<float> MeanPooling(DenseTensor<float> tokenTensor, DenseTensor<float> maskTensor)
        {
            var maskedSum = ApplyMaskAndSum(tokenTensor, maskTensor);
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
            var tokenTensor = new DenseTensor<T>(tokenShape);
            tokenEmbeddings.CopyTo(tokenTensor.Buffer.Span);
            return tokenTensor;
        }
    }
}
