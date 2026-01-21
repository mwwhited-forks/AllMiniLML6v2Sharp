using Microsoft.VisualStudio.TestTools.UnitTesting;
using OoBDev.AllMiniLmL6V2Sharp.Tokenizer;
using OoBDev.TestUtilities;

namespace OoBDev.AllMiniLmL6V2Sharp.Tests;

[TestClass]
public class TokenizerTests
{
    private const string vocabPath = "./model/vocab.txt";

    [DataTestMethod]
    [TestCategory(TestCategories.Simulate)]
    [DataRow("This is an example sentence")]
    [DataRow("Hello World!")]
    [DataRow("This is an example sentence.")]
    [DataRow("This is an example sentance")]
    [DataRow("sentance")]
    public void BertTokenizerTest(string sentence)
    {
        BertTokenizer tokenizer = new(vocabPath);
        IEnumerable<Token> tokenized = tokenizer.Tokenize(sentence);
        Assert.IsNotNull(tokenized);
        Assert.IsTrue(tokenized.Any());
    }
}
