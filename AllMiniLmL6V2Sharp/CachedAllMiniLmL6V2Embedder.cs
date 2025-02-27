using AllMiniLmL6V2Sharp.Tokenizer;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.IO;

namespace AllMiniLmL6V2Sharp;

public class CachedAllMiniLmL6V2Embedder : IEmbedder, IDisposable
{
    private readonly RunOptions _runOptions;
    private readonly InferenceSession _session;

    private readonly ITokenizer _tokenizer;
    private readonly string _modelPath;
    private readonly bool _truncate;
    /// <summary>
    /// Initializes the AllMiniLmL6v2 Embedder
    /// </summary>
    /// <param name="modelPath">Path to the embedding onnx model.</param>
    /// <param name="tokenizer">Optional custom tokenizer function.</param>
    /// <param name="truncate">If true, automatically truncates tokens to 512 tokens.</param>
    public CachedAllMiniLmL6V2Embedder(string modelPath = "./model/model.onnx", ITokenizer? tokenizer = null, bool truncate = false)
    {
        if (!Path.Exists(modelPath))
        {
            var tempPath = Path.Combine(Path.GetDirectoryName(this.GetType().Assembly.Location) ?? ".", modelPath);
            if (Path.Exists(tempPath)) modelPath = tempPath;
        }

        var vocabPath = Path.Combine(Path.GetDirectoryName(modelPath) ?? ".", "vocab.txt");

        _tokenizer = tokenizer ?? new BertTokenizer(vocabPath);
        _modelPath = modelPath;
        _truncate = truncate;

        _runOptions = new RunOptions();
        _session = new InferenceSession(_modelPath);
    }

    /// <summary>
    /// Generates an embedding array for the given sentence.
    /// </summary>
    /// <param name="sentence">Text to embed.</param>
    /// <returns>Sentence embeddings</returns>
    public IEnumerable<float> GenerateEmbedding(string sentence)
    {
        // Tokenize Input
        var tokens = _tokenizer.Tokenize(sentence);
        if (_truncate && tokens.Count() > 512)
        {
            tokens = tokens.Take(512);
        }

        var encodedTokens = _tokenizer.Encode(tokens.Count(), sentence);

        // Compute Token Embeddings
        var bertInput = new BertInput
        {
            InputIds = [.. encodedTokens.Select(t => t.InputIds)],
            TypeIds = [.. encodedTokens.Select(t => t.TokenTypeIds)],
            AttentionMask = [.. encodedTokens.Select(t => t.AttentionMask)]
        };

        // Create input tensors over the input data.
        using var inputIdsOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.InputIds,
              [1, bertInput.InputIds.Length]);

        using var attMaskOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.AttentionMask,
              [1, bertInput.AttentionMask.Length]);

        using var typeIdsOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.TypeIds,
              [1, bertInput.TypeIds.Length]);

        // Create input data for session. Request all outputs in this case.
        var inputs = new Dictionary<string, OrtValue>
        {
            { "input_ids", inputIdsOrtValue },
            { "attention_mask", attMaskOrtValue },
            { "token_type_ids", typeIdsOrtValue }
        };

        using var output = _session.Run(_runOptions, inputs, _session.OutputNames);

        // Perform Pooling
        var pooled = SingleMeanPooling(output[0], attMaskOrtValue);

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
    public IEnumerable<IEnumerable<float>> GenerateEmbeddings(IEnumerable<string> sentences) =>
        GenerateEmbeddingsInternal(sentences).ToArray();

    public async IAsyncEnumerable<(string sentence, float[] embedding)> GenerateEmbeddingsAsync(IEnumerable<string> sentences)
    {
        var tasks = sentences.AsParallel()
                .WithExecutionMode(ParallelExecutionMode.ForceParallelism)
                .Select(async sentence =>
                {
                    var embedding = GenerateEmbedding(sentence).ToArray();
                    await Task.Yield();
                    return (sentence, embedding);
                });

        var embeddings = await Task.WhenAll(tasks);

        foreach (var embedding in embeddings)
        {
            yield return embedding;
        }
    }

    private IEnumerable<IEnumerable<float>> GenerateEmbeddingsInternal(IEnumerable<string> sentences)
    {
        foreach (var sentence in sentences)
        {
            yield return GenerateEmbedding(sentence).ToArray();
        }
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
        var tokenShape = new ReadOnlySpan<int>([.. typeAndShape.Shape.Select(s => (int)s)]);
        var tokenEmbeddings = value.GetTensorDataAsSpan<T>();
        var tokenTensor = new DenseTensor<T>(tokenShape);
        tokenEmbeddings.CopyTo(tokenTensor.Buffer.Span);
        return tokenTensor;
    }

    public void Dispose()
    {
        // Dispose of unmanaged resources.
        Dispose(true);
        // Suppress finalization.
        GC.SuppressFinalize(this);
    }
    private bool _disposed = false;
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed)
        {
            return;
        }

        if (disposing)
        {
            _runOptions.Dispose();
            _session.Dispose();
        }

        _disposed = true;
    }

    ~CachedAllMiniLmL6V2Embedder() => Dispose(false);
}
