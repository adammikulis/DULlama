// using Microsoft.AspNetCore.Authentication.Negotiate;

// namespace DULlama
// {
//     public class Program
//     {
//         public static void Main(string[] args)
//         {
//             var builder = WebApplication.CreateBuilder(args);

//             // Add services to the container.

//             builder.Services.AddControllers();
//             // Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
//             builder.Services.AddEndpointsApiExplorer();
//             builder.Services.AddSwaggerGen();

//             builder.Services.AddAuthentication(NegotiateDefaults.AuthenticationScheme)
//                 .AddNegotiate();

//             builder.Services.AddAuthorization(options =>
//             {
//                 // By default, all incoming requests will be authorized according to the default policy.
//                 options.FallbackPolicy = options.DefaultPolicy;
//             });

//             var app = builder.Build();

//             // Configure the HTTP request pipeline.
//             if (app.Environment.IsDevelopment())
//             {
//                 app.UseSwagger();
//                 app.UseSwaggerUI();
//             }

//             app.UseHttpsRedirection();

//             app.UseAuthorization();


//             app.MapControllers();

//             app.Run();
//         }
//     }
// }

// This successfully generates embeddings
using LLama;
using LLama.Common;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;

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

DataTable dt = new DataTable();
dt.Columns.Add("Encoding", typeof(float[]));
dt.Columns.Add("OriginalText", typeof(string));

var model_name = "codellama";
var model_size = "7b";
var model_type = "instruct";
var model_quant = "Q4_K_M";
var full_model_name = model_name + "-" + model_size + "-" + model_type + "." + model_quant + ".gguf";
string modelPath = @"C:\ai\models\" + full_model_name;

var @modelparams = new ModelParams(modelPath)
{
    ContextSize = 4096,
    EmbeddingMode = true,
};

using var weights = LLamaWeights.LoadFromFile(@modelparams);
var embedder = new LLamaEmbedder(weights, @modelparams);

// Embed each fact and store it in the DataTable
foreach (var fact in du_facts)
{
    var embeddings = embedder.GetEmbeddings(fact);
    dt.Rows.Add(embeddings, fact);
}

Console.WriteLine("Facts embedded!");

// COMBINE THE BELOW CHAT SESSION WITH THE ABOVE CODE TO MAKE THE LLM LOOK AT THE ORIGINAL TEXT OF THE FACTS AND GIVE AN ANSWER TO THE USER BASED ON IT
using var context = weights.CreateContext(@modelparams);
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
        var factEmbeddings = (float[])row["Encoding"];
        var score = CosineSimilarity(queryEmbeddings, factEmbeddings);
        scores.Add(new Tuple<double, string>(score, (string)row["OriginalText"]));
    }

    // Get top n matches
    var n_top_matches = 3;
    var topMatches = scores.OrderByDescending(s => s.Item1).Take(n_top_matches).ToList();

    // Prepare prompt with original query and top n facts
    prompt = $"Reply in a conversational manner utilizing mainly the top facts in the prompt. Be a friendly but concise chatbot to help users learn more about the University of Denver. Query: {query}\n";
    for (int i = 0; i < topMatches.Count; i++)
    {
        prompt += $"Fact {i + 1}: {topMatches[i].Item2}\n";
    }
    prompt += "Answer:";
    conversation += prompt;
    

    Console.WriteLine("\nProcessing with LLM...");

    // Execute conversation with modified prompt including top n matches
    await foreach (var text in session.ChatAsync(new ChatHistory.Message(AuthorRole.User, prompt), new InferenceParams { Temperature = 0.5f, AntiPrompts = ["DU Llama: Please enter a query:\r\n"] }))
    {
        Console.Write(text);
    }
    prompt = "";
}