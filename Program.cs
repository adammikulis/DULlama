// This program creates a local RAG pipeline using LLamaSharp v0.9.1 utilizing Mistral or Llama2 transformer models
// It has a model directoryPath set to C:\ai\models, change to where you keep your downloaded models

// Download a Mistral math-enhanced model: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-DARE-GGUF
// Download a Mistral coding model: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-code-ft-GGUF
// Download a Mistral general instruct model: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
// 
// Download a Llama2 coding model: https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF
// Download a Llama2 general model: https://huggingface.co/TheBloke/Llama-2-7B-GGUF

// This project includes LLamaSharp.Backend.CPU only, for Nvidia GPUs add either backend from nuget:
// https://www.nuget.org/packages/LLamaSharp.Backend.Cuda11
// https://www.nuget.org/packages/LLamaSharp.Backend.Cuda12

using LLama;
using LLama.Common;
using System.Data;

string directoryPath = @"C:\ai\models"; // Change to your model folder here

// Create your datasource of facts for the vector db
string[] facts = {
    "The University of Denver is a private University that is abbreviated as 'DU'",
    "The University of Denver was founded in 1864",
    "DU is a private R1 University",
    "DU's Ritchie Center is home to the Magness Arena and Ritchie Center",
    "The mascot of the University of Denver is the Pioneer",
    "DU is located in south Denver, Colorado in the University neighborhood",
    "The 720 acre Kennedy Mountain Campus is located 110 miles northwest of Denver",
    "DU has 5700 undergraduate students and 7200 graduate students",
    "DU's hockey team plays in Magness Arena, named after cable television pioneer Bob Magness"
};

// This calculates how similar two vectors are
static double CosineSimilarity(float[] vector1, float[] vector2)
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

string modelPath = "";
string fullModelName = "";

// Determine if directory and models exist
if (!Directory.Exists(directoryPath)) 
{
    Console.WriteLine("The directory does not exist.");
    return;
}

string[] filePaths = Directory.GetFiles(directoryPath);
if (filePaths.Length == 0)
{
    Console.WriteLine("No models found in the directory");
    return;
}

bool validInput = false;

// Load a .gguf model from the selected folder
while (!validInput)
{
    // Display models names
    for (int i = 0; i < filePaths.Length; i++)
    {
        Console.WriteLine($"{i + 1}: {Path.GetFileName(filePaths[i])}");
    }

    Console.WriteLine("\nEnter the number of the model you want to load: ");
    if (int.TryParse(Console.ReadLine(), out int index) && index >= 1 && index <= filePaths.Length)
    {
        index -= 1;
        string selectedModelPath = filePaths[index];
        modelPath = selectedModelPath;
        fullModelName = Path.GetFileNameWithoutExtension(selectedModelPath);

        Console.WriteLine($"Model selected: {fullModelName}");
        validInput = true;
    }
    else{
        Console.WriteLine("Invalid input, please enter a number corresponding to the model list.\n");
    }
}

// Create model parameters to be used for inference and embedding
var @modelparams = new ModelParams(modelPath)
{
    ContextSize = 4096, // This can be changed by the user according to memory usage and model capability
    EmbeddingMode = true, // This must be set to true to generate embeddings for vector search
    // GpuLayerCount = 64 // Uncomment this line and set your number of layers to offload to the GPU here (must have nuget package LLamaSharp.Backend.Cuda11 or .Cuda12 installed)
};

// Load the model and create the embedder
using var model = LLamaWeights.LoadFromFile(@modelparams);
var embedder = new LLamaEmbedder(model, @modelparams);
Console.WriteLine($"\nModel: {fullModelName} loaded\n");

// Data table will include the embeddings (serves as the index) and original text for reference (atypical of vector dbs)
DataTable dt = new DataTable();
dt.Columns.Add("Embedding", typeof(float[]));
dt.Columns.Add("OriginalText", typeof(string));

// Uses the loaded model to embed each fact and store it in the DataTable
Console.WriteLine("\nEmbedding facts in vector database...\n");
foreach (var fact in facts)
{
    var embeddings = embedder.GetEmbeddings(fact);
    dt.Rows.Add(embeddings, fact);
}
Console.WriteLine("Facts embedded!\n");

// Create the context and InteractiveExecutor needed for chat, utilizing existing @modelparams
using var context = model.CreateContext(@modelparams);
var ex = new InteractiveExecutor(context);
string prompt = "";
string conversation = "";
ChatSession session = new ChatSession(ex);
Console.Write("\nDU Llama: Please enter a query:\r\n");

// Chat loop
while (true)
{
    // Reads the user query and generates embeddings
    var query = Console.ReadLine();
    if (string.IsNullOrWhiteSpace(query)) break; // Easy way to quit out
    var queryEmbeddings = embedder.GetEmbeddings(query);
    List<Tuple<double, string>> scores = new List<Tuple<double, string>>();

    // Compares embeddings to vector db and ranks by similarity
    foreach (DataRow row in dt.Rows)
    {
        var factEmbeddings = (float[])row["Embedding"];
        var score = CosineSimilarity(queryEmbeddings, factEmbeddings);
        scores.Add(new Tuple<double, string>(score, (string)row["OriginalText"]));
    }

    // Get top n matches from vector db
    var n_top_matches = 3;
    var topMatches = scores.OrderByDescending(s => s.Item1).Take(n_top_matches).ToList();

    // Prepare prompt with original query and top n facts
    prompt = $"Reply in a conversational manner utilizing the top facts in the prompt to answer only the user's specific question. Be a friendly but concise chatbot (do not offer extra, unrelated info) to help users learn more about the University of Denver. Query: {query}\n";
    for (int i = 0; i < topMatches.Count; i++)
    {
        prompt += $"Fact {i + 1}: {topMatches[i].Item2}\n";
    }
    prompt += "Answer:";

    // Execute conversation with modified prompt including top n matches
    Console.WriteLine("Processing with LLM...");
    await foreach (var text in session.ChatAsync(new ChatHistory.Message(AuthorRole.User, prompt), new InferenceParams { Temperature = 0.25f, AntiPrompts = ["DU Llama: Please enter a query:\r\n"] }))
    {
        Console.Write(text);
    }
    conversation += prompt; // Processing the full conversation is not yet implemented, treats each message as a new conversation at this time
    prompt = "";
}