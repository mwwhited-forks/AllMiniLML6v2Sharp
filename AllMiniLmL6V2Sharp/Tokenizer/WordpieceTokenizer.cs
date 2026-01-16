using System.Collections.Generic;
using System.Linq;

namespace AllMiniLmL6V2Sharp.Tokenizer;

internal class WordpieceTokenizer : BaseTokenizer
{
    private readonly IDictionary<string, int> _vocab;
    private readonly string _unknownToken;
    private readonly int _maxInputCharsPerWord;
    
    public WordpieceTokenizer(IDictionary<string, int> vocab, string unknownToken = Tokens.UNKNOWN_TOKEN, int maxInputCharsPerWord=200)
    {
        _vocab = vocab;
        _unknownToken = unknownToken;
        _maxInputCharsPerWord = maxInputCharsPerWord;
    }

    public override IEnumerable<string> Tokenize(string text)
    {
        List<string> output = new List<string>();
        foreach(var token in WhitespaceTokenize(text))
        {
            if (token.Length > _maxInputCharsPerWord)
            {
                output.Add(_unknownToken);
                continue;
            }

            var isBad = false;
            var start = 0;
            List<string> subTokens = new List<string>();
            while(start < token.Length)
            {
                var end = token.Length;
                string? currentSubstring = null;
                while(start < end)
                {
                    var substring = string.Join("", token.Skip(start).Take(end - start));
                    if(start > 0)
                    {
                        substring = "##" + substring;
                    }
                    if (_vocab.ContainsKey(substring))
                    {
                        currentSubstring = substring;
                        break;
                    }
                    end--;
                }
                if(currentSubstring == null)
                {
                    isBad = true;
                    break;
                }

                subTokens.Add(currentSubstring);
                start = end;
            }

            if (isBad)
            {
                output.Add(_unknownToken);
            }
            else
            {
                output.AddRange(subTokens);
            }
        }
        return output;
    }
}
