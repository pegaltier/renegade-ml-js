import init from "./dist/web/renegade.js";
import * as wasm from "./dist/web/renegade.js";
import { createAPI } from "./facade.js";

export const { Renegade, Neighbors } = createAPI(wasm);
export default init;
