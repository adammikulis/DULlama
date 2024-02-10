using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using LLama;
using LLama.Common;

public class RagPipelineBase
{
    protected string? directoryPath;
    protected string selectedModelPath;
    protected string? fullModelName;
    protected string? modelType;
    protected string[]? facts;
    protected uint? contextSize;
    protected DataTable dt;
    protected ModelParams? modelParams;
    protected LLamaEmbedder? embedder;
    protected InteractiveExecutor? executor;
    protected ChatSession? session;

    public event Action<string> OnMessage;

    public RagPipelineBase(string directoryPath, string[] facts, uint contextSize)
    {
        this.directoryPath = directoryPath;
        this.selectedModelPath = "";
        this.facts = facts;
        this.dt = new DataTable();
    }

    public virtual async Task InitializeAsync()
    {
        // Attempt to access provided directory path
        if (!Directory.Exists(directoryPath))
        {
            OnMessage?.Invoke("The directory does not exist.");
            return;
        }

        // Load paths of any models found
        var filePaths = Directory.GetFiles(directoryPath);
        if (filePaths.Length == 0)
        {
            OnMessage?.Invoke("No models found in the directory");
            return;
        }

        bool validModelSelected = false;

        // Loop for selecting which model to load
        while (!validModelSelected){
            // Display models names
            for (int i = 0; i < filePaths.Length; i++)
            {
                OnMessage?.Invoke($"{i + 1}: {Path.GetFileName(filePaths[i])}");
            }

            OnMessage?.Invoke("\nEnter the number of the model you want to load: ");
            if (int.TryParse(Console.ReadLine(), out int index) && index >= 1 && index <= filePaths.Length)
            {
                index -= 1;
                selectedModelPath = filePaths[index];
                fullModelName = Path.GetFileNameWithoutExtension(selectedModelPath);

                OnMessage?.Invoke($"Model selected: {fullModelName}");
                validModelSelected = true;

                // Determine the type of model based on the prefix of fullModelName
                modelType = fullModelName.Split('-')[0].ToLower();
            }
            
            else
            {
            OnMessage?.Invoke("Invalid input, please enter a number corresponding to the model list.\n");
            }
        }

        modelParams = new ModelParams(selectedModelPath)
        {
            ContextSize = 4096,
            EmbeddingMode = true,
        };

        using var model = LLamaWeights.LoadFromFile(modelParams);
        embedder = new LLamaEmbedder(model, modelParams);
        OnMessage?.Invoke($"Model: {fullModelName} from {selectedModelPath} loaded");

        InitializeDataTable();
    }

    protected void InitializeDataTable()
    {
        // Add columns for different types of embeddings and the original text
        dt.Columns.Add("LlamaEmbedding", typeof(float[]));
        dt.Columns.Add("MistralEmbedding", typeof(float[]));
        dt.Columns.Add("MixtralEmbedding", typeof(float[]));
        dt.Columns.Add("PhiEmbedding", typeof(float[]));
        dt.Columns.Add("OriginalText", typeof(string));

        OnMessage?.Invoke("Embedding facts in vector database...");

        // Embed facts and add them to the DataTable
        foreach (var fact in facts)
        {
            var embeddings = embedder.GetEmbeddings(fact);
            // Initialize embedding arrays to null
            float[]? llamaEmbedding = null;
            float[]? mistralEmbedding = null;
            float[]? mixtralEmbedding = null;
            float[]? phiEmbedding = null;

            // Assign embeddings based on the model type
            if (modelType == "llama")
            {
                llamaEmbedding = embeddings;
            }
            else if (modelType == "mistral")
            {
                mistralEmbedding = embeddings;
            }
            else if (modelType == "mixtral")
            {
                mixtralEmbedding = embeddings;
            }
            else if (modelType == "phi")
            {
                phiEmbedding = embeddings;
            }
            else
            {
                OnMessage?.Invoke($"Unsupported model type: {modelType}");
            }
        
            dt.Rows.Add(llamaEmbedding, mistralEmbedding, mixtralEmbedding, phiEmbedding, fact);
        }
        OnMessage?.Invoke("Facts embedded!");
    }
}

public class RagPipelineConsole : RagPipelineBase
{
    public RagPipelineConsole(string directoryPath, string[] facts, uint contextSize) : base(directoryPath, facts, contextSize)
    {
        this.OnMessage += Console.WriteLine;
    }
}
public static class VectorSearchUtility
{
    // Computes the cosine similarity between two vectors
    public static double CosineSimilarity(float[] vector1, float[] vector2)
    {
        double dotProduct = 0.0, magnitude1 = 0.0, magnitude2 = 0.0;
        int length = Math.Min(vector1.Length, vector2.Length);

        for (int i = 0; i < length; i++)
        {
            dotProduct += vector1[i] * vector2[i];
            magnitude1 += Math.Pow(vector1[i], 2);
            magnitude2 += Math.Pow(vector2[i], 2);
        }

        return dotProduct / (Math.Sqrt(magnitude1) * Math.Sqrt(magnitude2));
    }
}

