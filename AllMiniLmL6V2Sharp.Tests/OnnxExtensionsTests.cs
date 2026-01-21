using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using OoBDev.TestUtilities;

namespace OoBDev.AllMiniLmL6V2Sharp.Tests;

[TestClass]
public class OnnxExtensionsTests
{
    [TestMethod]
    [TestCategory(TestCategories.Simulate)]
    public void TestSum()
    {
        int[] dims = [1, 2, 4];
        DenseTensor<float> tensor = new(new float[] {1, 2, 3, 4, 5, 6, 7, 8}, dims);

        DenseTensor<float> summed = tensor.Sum(1);

        DenseTensor<float> expectedResult = new(new float[] { 6, 8, 10, 12 }, new int[] { 1, 4 });
        Assert.IsTrue(expectedResult.Dimensions.SequenceEqual(summed.Dimensions));
        Assert.IsTrue(expectedResult.SequenceEqual(summed));

    }

    [TestMethod]
    [TestCategory(TestCategories.Simulate)]
    public void TestUnsqueeze()
    {
        int[] dims = [1, 4];
        DenseTensor<float> tensor = new(new float[] { 1,1,1,1 }, dims);
        DenseTensor<float> expanded = tensor.Unsqueeze(-1);

        DenseTensor<float> expectedResult = new(new float[] { 1, 1, 1, 1 }, new int[] { 1, 4, 1 });
        Assert.IsTrue(expectedResult.Dimensions.SequenceEqual(expanded.Dimensions));
        Assert.IsTrue(expectedResult.SequenceEqual(expanded));
    }

    [TestMethod]
    [TestCategory(TestCategories.Simulate)]
    public void TestExpand()
    {
        int[] dims = [1, 4];
        DenseTensor<float> tensor = new(new float[] { 1, 1, 1, 1 }, dims);

        int[] expandDims = [1, 8];
        DenseTensor<float> expanded = tensor.Expand(expandDims);

        DenseTensor<float> expectedResult = new(new float[] { 1, 1, 1, 1, 1, 1, 1, 1 }, new int[] { 1, 8 });
        Assert.IsTrue(expectedResult.Dimensions.SequenceEqual(expanded.Dimensions));
        Assert.IsTrue(expectedResult.SequenceEqual(expanded));
    }

    [TestMethod]
    [TestCategory(TestCategories.Simulate)]
    public void TestElementWiseMultiply()
    {
        int[] dims = [1, 4];
        DenseTensor<float> x = new(new float[] { 1, 2, 3, 4 }, dims);
        DenseTensor<float> y = new(new float[] { 5, 6, 7, 8 }, dims);

        DenseTensor<float> result = x.ElementWiseMultiply(y);

        DenseTensor<float> expectedTensor = new(new float[] { 5, 12, 21, 32 }, dims);
        Assert.IsTrue(expectedTensor.Dimensions.SequenceEqual(result.Dimensions));
        Assert.IsTrue(result.SequenceEqual(expectedTensor));  
    }

    [TestMethod]
    [TestCategory(TestCategories.Simulate)]
    public void TestElementWiseDivide()
    {
        int[] dims = [1, 4];
        DenseTensor<float> x = new(new float[] { 1, 2, 3, 4 }, dims);
        DenseTensor<float> y = new(new float[] { 5, 6, 7, 8 }, dims);

        DenseTensor<float> result = x.ElementWiseDivide(y);

        DenseTensor<float> expectedTensor = new(new float[] { 0.2000f, 0.33333334f, 0.42857143f, 0.5000f }, dims);
        Assert.IsTrue(expectedTensor.Dimensions.SequenceEqual(result.Dimensions));
        Assert.IsTrue(result.SequenceEqual(expectedTensor));
    }

    [TestMethod]
    [TestCategory(TestCategories.Simulate)]
    public void TestClamp()
    {
        int[] dims = [1, 4];
        DenseTensor<float> x = new(new float[] { 1, 2, 3, 4 }, dims);

        DenseTensor<float> clammped = x.Clamp(min: 2);

        DenseTensor<float> expectedMin = new(new float[] { 2, 2, 3, 4 }, dims);
        Assert.IsTrue(expectedMin.Dimensions.SequenceEqual(clammped.Dimensions));
        Assert.IsTrue(expectedMin.SequenceEqual(clammped));

        DenseTensor<float> clammped2 = x.Clamp(max: 3);

        DenseTensor<float> expectedMax = new(new float[] { 1, 2, 3, 3 }, dims);
        Assert.IsTrue(expectedMax.Dimensions.SequenceEqual(clammped2.Dimensions));
        Assert.IsTrue(expectedMax.SequenceEqual(clammped2));

        DenseTensor<float> clammped3 = x.Clamp(2, 3);

        DenseTensor<float> expectedBoth = new(new float[] { 2, 2, 3, 3 }, dims);
        Assert.IsTrue(expectedBoth.Dimensions.SequenceEqual(clammped3.Dimensions));
        Assert.IsTrue(expectedBoth.SequenceEqual(clammped3));
    }

    [TestMethod]
    [TestCategory(TestCategories.Simulate)]
    public void TestNormalize()
    {
        DenseTensor<float> value = new(new float[] { 1, 2, 3, 4 }, new int[] { 1, 4 });
        DenseTensor<float> normalized = value.Normalize(p: 2, dim: 1);

        DenseTensor<float> expected = new(new float[] { 0.18257418f, 0.36514837f, 0.5477225f, 0.73029673f }, new int[] { 1, 4 });
        Assert.IsTrue(normalized.Dimensions.SequenceEqual(expected.Dimensions));
        Assert.IsTrue(normalized.SequenceEqual(expected));
    }
}
