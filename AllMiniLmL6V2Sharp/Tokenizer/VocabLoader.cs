using System.Collections.Generic;
using System.IO;

namespace AllMiniLmL6V2Sharp.Tokenizer;

public class VocabLoader
{
    public static IDictionary<string, int> Load(string path)
    {
        IDictionary<string, int> vocab = new Dictionary<string, int>();
        var index = 0;
        IEnumerable<string> lines = File.ReadLines(path);
        foreach (var line in lines)
        {
            if(string.IsNullOrEmpty(line)) break;
            var trimmedLine = line.Trim();
            vocab.Add(trimmedLine, index++);
        }

        return vocab;
    }
}
