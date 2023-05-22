import * as dotenv from "dotenv";
dotenv.config();
import readline from "readline";
import chalk from "chalk";
import ora from "ora";
import pkg from "enquirer";
import { globIterate } from "glob";
const { prompt } = pkg;
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter, MarkdownTextSplitter, } from "langchain/text_splitter";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { Command } from "@commander-js/extra-typings";
const program = new Command()
    .option("-i, --input-file <pattern>", "Text file(s) to ingest into GPT embeddings vectorstore (accepts wildcards).")
    .requiredOption("-d, --data-store <dir>", "Directory to store vectorstore data.")
    .requiredOption("-t, --input-type <type>", "Input type. Options are 'text', 'markdown' or 'pdf'.")
    .option("-n, --new-vector-store", "Create a new vectorstore");
program.parse(process.argv);
const options = program.opts();
const spinner = ora(`Loading text..`).start();
let files;
if (options.inputFile) {
    files = globIterate(options.inputFile);
}
else {
    files = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
        terminal: false,
    });
}
for await (const line of files) {
    let loader;
    let splitter;
    switch (options.inputType) {
        case "text":
            loader = new TextLoader(line);
            splitter = new RecursiveCharacterTextSplitter();
            break;
        case "markdown":
            loader = new TextLoader(line);
            splitter = new MarkdownTextSplitter();
            break;
        case "pdf":
            loader = new PDFLoader(line, { splitPages: false });
            splitter = new RecursiveCharacterTextSplitter();
    }
    if (!loader || !splitter) {
        throw new Error(`Invalid input type ${options.inputType}`);
    }
    const rawDocs = await loader.load();
    spinner.succeed(`Loaded ${chalk.green(line)}.`);
    spinner.start(`Splitting text into chunks..`);
    const docs = await splitter.splitDocuments(rawDocs);
    spinner.succeed();
    let vectorStore;
    if (options.newVectorStore) {
        const answer = await prompt({
            type: "confirm",
            name: "question",
            message: "Creating a new vectorsotre will overwrite any data in an existing directory. Are you sure?",
        });
        if (!answer) {
            process.exit(0);
        }
        vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings({ openAIApiKey: process.env.OPENAI_API_KEY }));
    }
    else {
        vectorStore = await HNSWLib.load(options.dataStore, new OpenAIEmbeddings({ openAIApiKey: process.env.OPENAI_API_KEY }));
        await vectorStore.addDocuments(docs);
    }
    spinner.start(`Ingesting text into vectorstore..`);
    await vectorStore.save(options.dataStore);
    spinner.succeed(`Ingested ${chalk.green(docs.length)} documents into vectorstore ${chalk.green(options.dataStore)}.`);
}
spinner.succeed(chalk.green(`Finished.`));
