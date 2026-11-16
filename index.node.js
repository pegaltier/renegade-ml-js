import { createRequire } from "node:module";
import { createAPI } from "./facade.js";

const require = createRequire(import.meta.url);
const wasm = require("./dist/node/renegade.js");
export const { Renegade, Neighbors } = createAPI(wasm);
export default async function init() {}
