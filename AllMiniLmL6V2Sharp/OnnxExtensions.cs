using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Linq;

/// <summary>
/// Extension methods for ONNX tensor operations.
/// </summary>
public static class OnnxExtensions
{
    /// <summary>
    /// Inserts a new axis of size 1 at the specified position.
    /// </summary>
    /// <typeparam name="T">The tensor element type.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="axis">The axis position to insert the new dimension.</param>
    /// <returns>A new tensor with an additional dimension of size 1.</returns>
    public static DenseTensor<T> Unsqueeze<T>(this DenseTensor<T> tensor, int axis)
    {
        var originalShape = tensor.Dimensions;
        var newData = tensor.ToArray();

        var newShape = originalShape.ToArray().ToList();
        var newAxis = axis + originalShape.Length + 1;
        newShape.Insert(newAxis, 1);

        var unsqueezedTensor = new DenseTensor<T>(newData, newShape.ToArray());

        return unsqueezedTensor;
    }

    /// <summary>
    /// Expands the input tensor to a new shape by broadcasting dimensions.
    /// </summary>
    /// <typeparam name="T">The tensor element type.</typeparam>
    /// <param name="input">The input tensor.</param>
    /// <param name="newShape">The target shape for expansion.</param>
    /// <returns>A new tensor with the expanded shape.</returns>
    public static DenseTensor<T> Expand<T>(this DenseTensor<T> input, int[] newShape)
    {
        var inputData = input.ToArray();
        var originalShape = input.Dimensions.ToArray();

        if (newShape.Length != originalShape.Length)
        {
            throw new ArgumentException("The length of newShape must be equal to the number of dimensions in the original tensor.");
        }

        var expandedData = new T[newShape.Aggregate((a, b) => a * b)];
        var index = new int[newShape.Length];
        var strides = new int[originalShape.Length];

        // Calculate the strides
        strides[originalShape.Length - 1] = 1;
        for (var i = originalShape.Length - 2; i >= 0; i--)
        {
            strides[i] = strides[i + 1] * originalShape[i + 1];
        }

        // Perform the expansion
        for (var i = 0; i < expandedData.Length; i++)
        {
            expandedData[i] = inputData[
                Enumerable.Range(0, originalShape.Length)
                          .Select(d => index[d] % originalShape[d])
                          .Select((d, j) => d * strides[j])
                          .Sum()
            ];

            // Update the index for the next iteration
            for (var j = 0; j < index.Length; j++)
            {
                index[j]++;
                if (index[j] == newShape[j])
                {
                    index[j] = 0;
                }
                else
                {
                    break;
                }
            }
        }

        return new DenseTensor<T>(expandedData, newShape.ToArray());
    }

    /// <summary>
    /// Performs element-wise multiplication of two tensors.
    /// </summary>
    /// <param name="tensor1">The first tensor.</param>
    /// <param name="tensor2">The second tensor.</param>
    /// <returns>A new tensor containing the element-wise product.</returns>
    public static DenseTensor<float> ElementWiseMultiply(this DenseTensor<float> tensor1, DenseTensor<float> tensor2)
    {
        var resultData = new float[tensor1.Length];

        for (var i = 0; i < resultData.Length; i++)
        {
            resultData[i] = tensor1.Buffer.Span[i] * tensor2.Buffer.Span[i];
        }

        return new DenseTensor<float>(resultData, tensor1.Dimensions);
    }

    /// <summary>
    /// Computes the sum of tensor elements over specified dimensions.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="dim">The dimension to sum along. If null, sums all elements.</param>
    /// <param name="keepdim">Whether to keep the reduced dimension in the output.</param>
    /// <returns>A new tensor containing the sum.</returns>
    public static DenseTensor<float> Sum(this DenseTensor<float> input, int? dim = null, bool keepdim = false)
    {
        if (dim == null || !dim.HasValue)
        {
            // If dim is null, sum across all dimensions
            return SumAll(input, keepdim);
        }
        else
        {
            // Sum along the specified dimension
            return SumAlongDimension(input, dim.Value, keepdim);
        }
    }

    private static DenseTensor<float> SumAlongDimension(DenseTensor<float> inputTensor, int dim, bool keepdim)
    {
        var dimensions = inputTensor.Dimensions.ToArray();
        var outputShape = keepdim ? dimensions : [.. dimensions.Select((index, idx) => dim == idx ? -1 : index).Where(x => x != -1)];
        var result = new float[outputShape.Aggregate((a, b) => a * b)];

        SumAlongDimensionRecursive(inputTensor, dim, keepdim, dimensions, result, new int[dimensions.Length], 0);

        return new DenseTensor<float>(result, outputShape);
    }

    private static void SumAlongDimensionRecursive(DenseTensor<float> inputTensor, int dim, bool keepdim, int[] dimensions, float[] result, int[] indices, int depth)
    {
        if (depth == dimensions.Length)
        {
            var element = (float)inputTensor.GetValue(GetFlattenedIndex(indices, dimensions));
            int[] reducedIndices = [.. indices.Select((index, idx) => dim == idx ? 0 : index)];
            var outputIndices = keepdim
                ? reducedIndices
                : [.. reducedIndices.Where((index, idx) => dimensions[idx] != 1)];
            var resultIndex = GetFlattenedIndex(outputIndices, dimensions);
            result[resultIndex] = result[resultIndex] +  element;
        }
        else
        {
            for (var i = 0; i < dimensions[depth]; i++)
            {
                indices[depth] = i;
                SumAlongDimensionRecursive(inputTensor, dim, keepdim, dimensions, result, indices, depth + 1);
            }
        }
    }

    private static DenseTensor<float> SumAll(DenseTensor<float> input, bool keepdim)
    {
        var sumResult = new float[1];
        for (var i = 0; i < input.Buffer.Span.Length; i++)
        {
            sumResult[0] += input.Buffer.Span[i];
        }

        return keepdim ? new DenseTensor<float>(sumResult, new int[] { 1 }) : new DenseTensor<float>(sumResult, new int[] { });
    }

    private static int GetFlattenedIndex(int[] indices, int[] dimensions)
    {
        var index = 0;
        var multiplier = 1;

        for (var i = indices.Length - 1; i >= 0; i--)
        {
            index += indices[i] * multiplier;
            multiplier *= dimensions[i];
        }

        return index;
    }

    /// <summary>
    /// Clamps all elements in the tensor to be within a specified range.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="min">The minimum value.</param>
    /// <param name="max">The maximum value.</param>
    /// <returns>A new tensor with values clamped to the specified range.</returns>
    public static DenseTensor<float> Clamp(this DenseTensor<float> input, float min = float.MinValue, float max = float.MaxValue)
    {
        var resultData = new float[input.Length];

        for (var i = 0; i < resultData.Length; i++)
        {
            resultData[i] = Math.Min(Math.Max(input.Buffer.Span[i], min), max);
        }

        return new DenseTensor<float>(resultData, input.Dimensions);
    }

    /// <summary>
    /// Performs element-wise division of two tensors.
    /// </summary>
    /// <param name="tensor1">The numerator tensor.</param>
    /// <param name="tensor2">The denominator tensor.</param>
    /// <returns>A new tensor containing the element-wise quotient.</returns>
    public static DenseTensor<float> ElementWiseDivide(this DenseTensor<float> tensor1, DenseTensor<float> tensor2)
    {
        var resultData = new float[tensor1.Length];

        for (var i = 0; i < resultData.Length; i++)
        {
            resultData[i] = tensor1.Buffer.Span[i] / tensor2.Buffer.Span[i];
        }

        return new DenseTensor<float>(resultData, tensor1.Dimensions);
    }

    /// <summary>
    /// Normalizes the tensor along the specified dimension using the p-norm.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="p">The norm degree (e.g., 2 for L2 normalization).</param>
    /// <param name="dim">The dimension along which to normalize.</param>
    /// <returns>A new normalized tensor.</returns>
    public static DenseTensor<float> Normalize(this DenseTensor<float> input, int p, int dim)
    {
        var normalizedData = new float[input.Length];

        for (var i = 0; i < input.Length; i++)
        {
            normalizedData[i] = input.Buffer.Span[i] / Norm(input, p, dim, i);
        }

        return new DenseTensor<float>(normalizedData, input.Dimensions);
    }

    private static float Norm(DenseTensor<float> input, int p, int dim, int flatIndex)
    {
        var indices = GetIndex(flatIndex, input.Dimensions.ToArray(), input.Buffer.Span.Length);
        var sum = 0.0f;

        for (var i = 0; i < input.Dimensions[dim]; i++)
        {
            indices[dim] = i;
            sum += (float)Math.Pow(input[indices], p);
        }

        return (float)Math.Pow(sum, 1.0 / p);
    }

    private static int[] GetIndex(int index, int[] dimensions, int mul)
    {
        var res = new int[dimensions.Length];

        for (var i = dimensions.Length; i != 0; --i)
        {
            mul /= dimensions[i - 1];
            res[i - 1] = index / mul;
            if (res[i - 1] >= dimensions[i - 1]) throw new Exception("Invalid Index");
            index -= res[i - 1] * mul;
        }
        return res;
    }
}