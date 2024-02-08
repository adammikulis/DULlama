// This program creates a local RAG pipeline using Mistral or Llama
using LLama;
using LLama.Common;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;

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

// Model path and params
var model_name = "mistral";
var model_size = "7b";
var model_type = "instruct-v0.2-code-ft";
var model_quant = "Q2_K";
var full_model_name = model_name + "-" + model_size + "-" + model_type + "." + model_quant + ".gguf";
string modelPath = @"C:\ai\models\" + full_model_name;

var @modelparams = new ModelParams(modelPath)
{
    ContextSize = 4096, // This can be changed by the user according to memory usage and model capability
    EmbeddingMode = true, // This must be set to true to generate embeddings for vector search
};

// Load the model and create the embedder
using var model = LLamaWeights.LoadFromFile(@modelparams);
var embedder = new LLamaEmbedder(model, @modelparams);

// Create the datasource for the vector db
string[] du_facts = {
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

// Data table includes the embedding and original text for reference (atypical of vector dbs)
DataTable dt = new DataTable();
dt.Columns.Add("Embedding", typeof(float[]));
dt.Columns.Add("OriginalText", typeof(string));

// Embed each fact and store it in the DataTable
foreach (var fact in du_facts)
{
    var embeddings = embedder.GetEmbeddings(fact);
    dt.Rows.Add(embeddings, fact);
}
Console.WriteLine("Facts embedded!");

using var context = model.CreateContext(@modelparams);
var ex = new InteractiveExecutor(context);
string prompt = "";
string conversation = "";

ChatSession session = new ChatSession(ex);
Console.Write("DU Llama: Please enter a query:\r\n");
while (true)
{
    var query = Console.ReadLine();

    if (string.IsNullOrWhiteSpace(query)) break;

    var queryEmbeddings = embedder.GetEmbeddings(query);
    List<Tuple<double, string>> scores = new List<Tuple<double, string>>();

    foreach (DataRow row in dt.Rows)
    {
        var factEmbeddings = (float[])row["Embedding"];
        var score = CosineSimilarity(queryEmbeddings, factEmbeddings);
        scores.Add(new Tuple<double, string>(score, (string)row["OriginalText"]));
    }

    // Get top n matches
    var n_top_matches = 3;
    var topMatches = scores.OrderByDescending(s => s.Item1).Take(n_top_matches).ToList();

    // Prepare prompt with original query and top n facts
    prompt = $"Reply in a conversational manner utilizing mainly the top facts in the prompt to answer only the user's specific question. Be a friendly but concise chatbot (do not offer extra, less related info) to help users learn more about the University of Denver. Query: {query}\n";
    for (int i = 0; i < topMatches.Count; i++)
    {
        prompt += $"Fact {i + 1}: {topMatches[i].Item2}\n";
    }
    prompt += "Answer:";
    conversation += prompt;
    

    Console.WriteLine("\nProcessing with LLM...");

    // Execute conversation with modified prompt including top n matches
    await foreach (var text in session.ChatAsync(new ChatHistory.Message(AuthorRole.User, prompt), new InferenceParams { Temperature = 0.25f, AntiPrompts = ["DU Llama: Please enter a query:\r\n"] }))
    {
        Console.Write(text);
    }
    prompt = "";
}