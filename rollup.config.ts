import typescript from "@rollup/plugin-typescript";
import { terser } from "rollup-plugin-terser";
import nodeResolve from "@rollup/plugin-node-resolve";
import commonjs from "@rollup/plugin-commonjs";

const production = !process.env.ROLLUP_WATCH;

export default () => {
    const commonJSBuild = {
        input: "src/main.ts",
        output: {
            file: "dist/main.js",
            format: "cjs", //common js output
        },
        plugins: [
            nodeResolve({ exportConditions: ["node"] }),
            commonjs(),
            typescript(),
            production && terser(),
        ],
    };

    const esmBuild = {
        ...commonJSBuild,
        output: {
            file: "dist/es/main.js",
            format: "esm",
        },
    };
    return [commonJSBuild, esmBuild];
};
