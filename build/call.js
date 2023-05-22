import dotenv from "dotenv";
dotenv.config();
import fs from "fs";
import readline from "readline";
import chalk from "chalk";
import ora from "ora";
import { OpenAI } from "langchain/llms/openai";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { VectorStoreToolkit, createVectorStoreAgent } from "langchain/agents";
import { Command } from "@commander-js/extra-typings";
const program = new Command()
    .option("-i, --input-file <file>", "Input values to send to the vectorstore agent in JSONL format.")
    .requiredOption("-v, --vector-store <dir>", "Directory to store vectorstore data.")
    .requiredOption("-n, --vector-store-name <name>", "Name of the vectorstore.")
    .requiredOption("-d, --vector-store-description <description>", "Description of the vectorstore.")
    .option("-o, --output-file <file>", "Chain values to output in JSONL format.");
program.parse(process.argv);
const options = program.opts();
const vectorStore = await HNSWLib.load(options.vectorStore, new OpenAIEmbeddings({ openAIApiKey: process.env.OPENAI_API_KEY }));
const vectorStoreInfo = {
    name: options.vectorStoreName,
    description: options.vectorStoreDescription,
    vectorStore,
};
const model = new OpenAI({ temperature: 0 });
const toolkit = new VectorStoreToolkit(vectorStoreInfo, model);
const agent = createVectorStoreAgent(model, toolkit);
const output = options.outputFile
    ? fs.createWriteStream(options.outputFile)
    : undefined;
const spinner = ora(`Loading inputs..`).start();
const rl = readline.createInterface({
    input: options.inputFile
        ? fs.createReadStream(options.inputFile)
        : process.stdin,
    output: process.stdout,
    terminal: false,
});
spinner.succeed();
spinner.start(`Send inputs to the agent.`);
let count = 0;
for await (const line of rl) {
    const input = JSON.parse(line);
    const result = await agent.call(input);
    output
        ? output.write(JSON.stringify(result) + "\n")
        : process.stdout.write(JSON.stringify(result) + "\n");
    spinner.suffixText = chalk.dim(`Processed ${++count} inputs.`);
}
spinner.succeed(chalk.green(`Finished.`));
