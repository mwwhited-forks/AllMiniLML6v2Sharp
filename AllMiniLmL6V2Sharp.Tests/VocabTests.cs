using Microsoft.VisualStudio.TestTools.UnitTesting;
using OoBDev.AllMiniLmL6V2Sharp.Tokenizer;
using OoBDev.TestUtilities;

namespace OoBDev.AllMiniLmL6V2Sharp.Tests;

[TestClass]
public class VocabTests
{
    private const string vocabPath = "./model/vocab.txt";

    [TestMethod]
    [TestCategory(TestCategories.Simulate)]
    public void LoadVocabTest()
    {
        var vocab = VocabLoader.Load(vocabPath);
        Assert.IsNotNull(vocab);
        Assert.IsTrue(vocab.Count > 0);
    }
}
