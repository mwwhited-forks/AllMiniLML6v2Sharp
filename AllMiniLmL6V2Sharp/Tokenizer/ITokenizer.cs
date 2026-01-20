using System.Collections.Generic;

namespace OoBDev.AllMiniLmL6V2Sharp.Tokenizer;

public interface ITokenizer
{
    IEnumerable<Token> Tokenize(string text);
    IEnumerable<EncodedToken> Encode(int sequenceLength, string text);

}
