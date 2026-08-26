import { readFileSync } from "node:fs";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";

const dashboardRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const distRoot = resolve(dashboardRoot, "dist");
const manifestPath = resolve(distRoot, ".vite", "manifest.json");
const budgetPath = resolve(
    process.env.AVALANCHE_BUNDLE_BUDGET ?? resolve(dashboardRoot, "bundle-budget.json"),
);

function readJson(path, name) {
    try {
        return JSON.parse(readFileSync(path, "utf8"));
    } catch (error) {
        throw new Error(`the ${name} is missing or damaged`, { cause: error });
    }
}

function loadBudget() {
    const budget = readJson(budgetPath, "bundle budget");
    const valid =
        budget?.budget_version === 1
        && typeof budget.entry_source === "string"
        && Number.isInteger(budget.reference_raw_bytes)
        && Number.isInteger(budget.reference_gzip_bytes)
        && Number.isInteger(budget.maximum_raw_bytes)
        && budget.reference_raw_bytes > 0
        && budget.reference_gzip_bytes > 0
        && budget.maximum_raw_bytes >= budget.reference_raw_bytes;
    if (!valid) throw new Error("the bundle budget is invalid");
    return budget;
}

function loadChunks(manifest, entrySource) {
    if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
        throw new Error("the Vite manifest is invalid");
    }
    const entries = Object.entries(manifest).filter(
        ([, chunk]) => chunk && typeof chunk === "object" && chunk.isEntry === true,
    );
    const selected = entries.filter(([, chunk]) => chunk.src === entrySource);
    if (selected.length !== 1) {
        throw new Error("the Vite manifest needs one configured entry");
    }
    const visited = new Set();
    const visit = (key) => {
        if (visited.has(key)) return;
        const chunk = manifest[key];
        if (!chunk || typeof chunk !== "object" || typeof chunk.file !== "string") {
            throw new Error("the Vite manifest contains a damaged chunk");
        }
        if (chunk.imports !== undefined && !Array.isArray(chunk.imports)) {
            throw new Error("the Vite manifest contains damaged imports");
        }
        visited.add(key);
        for (const dependency of chunk.imports ?? []) {
            if (typeof dependency !== "string") {
                throw new Error("the Vite manifest contains a damaged import");
            }
            visit(dependency);
        }
    };
    visit(selected[0][0]);
    return [...visited].map((key) => manifest[key].file).sort();
}

function readChunk(file) {
    const path = resolve(distRoot, file);
    const outside = relative(distRoot, path).startsWith(`..${sep}`) || isAbsolute(relative(distRoot, path));
    if (outside) throw new Error("a bundle chunk leaves the build directory");
    try {
        const content = readFileSync(path);
        return {
            file,
            rawBytes: content.byteLength,
            gzipBytes: gzipSync(content).byteLength,
        };
    } catch (error) {
        throw new Error(`the bundle chunk ${file} is missing`, { cause: error });
    }
}

function main() {
    const budget = loadBudget();
    const manifest = readJson(manifestPath, "Vite manifest");
    const chunks = loadChunks(manifest, budget.entry_source).map(readChunk);
    const rawBytes = chunks.reduce((total, chunk) => total + chunk.rawBytes, 0);
    const gzipBytes = chunks.reduce((total, chunk) => total + chunk.gzipBytes, 0);
    console.log("Initial static chunks:");
    for (const chunk of chunks) {
        console.log(`- ${chunk.file}: ${chunk.rawBytes} raw, ${chunk.gzipBytes} gzip`);
    }
    console.log(`Initial static total: ${rawBytes} raw, ${gzipBytes} gzip`);
    if (rawBytes > budget.maximum_raw_bytes) {
        throw new Error(
            `the initial bundle exceeds ${budget.maximum_raw_bytes} raw bytes`,
        );
    }
}

try {
    main();
} catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
}
