const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const WasmPackPlugin = require("@wasm-tool/wasm-pack-plugin");

module.exports = (_, argv) => {
    console.log('Building in %s mode', argv.mode);
    config = {
        entry: './index.js',
        output: {
            path: path.resolve(__dirname, 'dist'),
            filename: 'index.js',
        },
        plugins: [
            new HtmlWebpackPlugin({
                template: 'index.html'
            }),
            new WasmPackPlugin({
                extraArgs: "-- --features=build-wasm",
                crateDirectory: path.resolve(__dirname, ".")
            })
        ],
        experiments: {
            asyncWebAssembly: true
        }
    };
    return config;
}