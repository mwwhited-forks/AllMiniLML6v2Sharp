﻿using System.Numerics.Tensors;
using Microsoft.ML.OnnxRuntime;

namespace AllMiniLmL6V2Sharp.Tests
{
    public class CachedAllMiniLmL6V2Tests
    {
        [Fact]
        public void SameSameTest()
        {
            var model = new AllMiniLmL6V2Embedder();
            using var cachedModel = new CachedAllMiniLmL6V2Embedder();

            var sentence = "This is an example sentence";
            var embedding = model.GenerateEmbedding(sentence);
            Assert.NotNull(embedding);
            Assert.NotEmpty(embedding);

            var cachedEmbedding = cachedModel.GenerateEmbedding(sentence);
            Assert.NotNull(embedding);
            Assert.NotEmpty(embedding);

            var similarity = TensorPrimitives.CosineSimilarity(embedding.ToArray(), cachedEmbedding.ToArray());
            Assert.Equal(1d, similarity);
        }


        [Fact]
        public void ModelTest()
        {
            using var model = new CachedAllMiniLmL6V2Embedder();
            var sentence = "This is an example sentence";
            var embedding = model.GenerateEmbedding(sentence);
            Assert.NotNull(embedding);
            Assert.NotEmpty(embedding);
        }

        [Fact]
        public void ModelMultipleTest()
        {
            using var model = new CachedAllMiniLmL6V2Embedder();
            string[] sentences = ["This is an example sentence", "Here is another"];
            var embedding = model.GenerateEmbeddings(sentences);
            Assert.NotNull(embedding);
            Assert.NotEmpty(embedding);
        }

        [Fact]
        public void LongContextTest()
        {
            using var model = new CachedAllMiniLmL6V2Embedder();
            string[] sentences = ["Lorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur ac mauris nulla. Nullam rutrum, urna eu elementum cursus, eros risus sollicitudin magna, maximus eleifend mi lorem et tellus. Fusce lacus tellus, consectetur ac turpis in, ultricies consequat felis. Aliquam imperdiet tristique ante at consequat. Fusce sed efficitur ipsum. Aliquam erat volutpat. In molestie sapien non porttitor iaculis. Nunc pretium mauris nisl, eu luctus augue volutpat dapibus. Ut rutrum nec ante vitae commodo. Praesent facilisis eget leo eget congue.\r\n\r\nUt luctus lorem ut finibus sollicitudin. Morbi nec elit vel lorem congue condimentum. Vivamus feugiat enim et sapien mollis, nec feugiat lacus pellentesque. Cras aliquet, nisi at imperdiet dictum, metus ex congue elit, sed semper est erat imperdiet leo. Curabitur consequat urna turpis, a sollicitudin justo eleifend ac. Nunc dignissim tincidunt erat vitae ullamcorper. Phasellus nulla urna, gravida ac neque et, rhoncus convallis lorem. Etiam id erat sit amet leo mollis viverra. Pellentesque efficitur pretium nunc.\r\n\r\nCurabitur eget est metus. Donec mollis, tortor eu finibus volutpat, odio ex scelerisque dui, id lobortis neque est ac dolor. Praesent bibendum ex vel ultricies varius. Nullam molestie massa vitae nunc faucibus, id pellentesque augue tincidunt. Morbi fermentum dolor quis gravida venenatis. Duis mattis in nisl eu fringilla. Proin vulputate dui vel ligula rhoncus lobortis.\r\n\r\n"];
            Assert.Throws<OnnxRuntimeException>(() => model.GenerateEmbeddings(sentences));
        }

        [Fact]
        public void TruncateTest()
        {
            using var model = new CachedAllMiniLmL6V2Embedder(truncate: true);
            string[] sentences = ["Lorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur ac mauris nulla. Nullam rutrum, urna eu elementum cursus, eros risus sollicitudin magna, maximus eleifend mi lorem et tellus. Fusce lacus tellus, consectetur ac turpis in, ultricies consequat felis. Aliquam imperdiet tristique ante at consequat. Fusce sed efficitur ipsum. Aliquam erat volutpat. In molestie sapien non porttitor iaculis. Nunc pretium mauris nisl, eu luctus augue volutpat dapibus. Ut rutrum nec ante vitae commodo. Praesent facilisis eget leo eget congue.\r\n\r\nUt luctus lorem ut finibus sollicitudin. Morbi nec elit vel lorem congue condimentum. Vivamus feugiat enim et sapien mollis, nec feugiat lacus pellentesque. Cras aliquet, nisi at imperdiet dictum, metus ex congue elit, sed semper est erat imperdiet leo. Curabitur consequat urna turpis, a sollicitudin justo eleifend ac. Nunc dignissim tincidunt erat vitae ullamcorper. Phasellus nulla urna, gravida ac neque et, rhoncus convallis lorem. Etiam id erat sit amet leo mollis viverra. Pellentesque efficitur pretium nunc.\r\n\r\nCurabitur eget est metus. Donec mollis, tortor eu finibus volutpat, odio ex scelerisque dui, id lobortis neque est ac dolor. Praesent bibendum ex vel ultricies varius. Nullam molestie massa vitae nunc faucibus, id pellentesque augue tincidunt. Morbi fermentum dolor quis gravida venenatis. Duis mattis in nisl eu fringilla. Proin vulputate dui vel ligula rhoncus lobortis.\r\n\r\n"];
            var embedding = model.GenerateEmbeddings(sentences);
            Assert.NotNull(embedding);
            Assert.NotEmpty(embedding);
        }

        [Fact]
        public void MultipleSameTest()
        {
            using var model = new CachedAllMiniLmL6V2Embedder();
            string sentence = "This is an example sentence";
            string[] sentences = ["other", sentence, sentence, sentence];
            var embedding = model.GenerateEmbedding(sentence);
            var embeddings = model.GenerateEmbeddings(sentences);
            Assert.NotNull(embedding);
            Assert.NotEmpty(embedding);
            Assert.NotNull(embeddings);
            Assert.NotEmpty(embeddings);
            Assert.Equal(sentences.Length, embeddings.Count());

            // Make sure all embeddings are the same.
            Assert.True(embeddings.Skip(1).All(e => e.SequenceEqual(embeddings.Skip(1).First())));
            Assert.True(embeddings.Skip(1).All(e => e.SequenceEqual(embedding)));

            foreach (var batchEmbedding in embeddings.Skip(1))
            {
                Assert.True(batchEmbedding.SequenceEqual(embedding),
                    "Single embedding does not match a batch embedding");
            }
        }

        [Fact]
        public async Task BigTest()
        {
            using var model = new CachedAllMiniLmL6V2Embedder();

            var inputs = Enumerable.Range(0, 200).Select(i => Guid.NewGuid().ToString());

            var tasks = inputs.AsParallel().Select(async sentence =>
            {
                await Task.Yield();
                var embedding = model.GenerateEmbedding(sentence);
                return (sentence, embedding);
            });

            var results = await Task.WhenAll(tasks);
        }
    }
}