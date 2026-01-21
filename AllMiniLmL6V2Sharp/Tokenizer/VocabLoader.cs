using System.Collections.Generic;
using System.IO;

namespace OoBDev.AllMiniLmL6V2Sharp.Tokenizer;

/// <summary>
/// Utility class for loading BERT vocabulary files.
/// </summary>
public class VocabLoader
{
    /// <summary>
    /// Loads a vocabulary file into a dictionary.
    /// </summary>
    /// <param name="path">Path to the vocabulary file.</param>
    /// <returns>A dictionary mapping tokens to their indices.</returns>
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
