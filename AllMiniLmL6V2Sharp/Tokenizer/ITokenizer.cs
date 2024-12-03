using System.Collections.Generic;

namespace AllMiniLmL6V2Sharp.Tokenizer
{
    public interface ITokenizer
    {
        Token[] Tokenize(string text);
        EncodedToken[] Encode(int sequenceLength, string text);

    }
}
