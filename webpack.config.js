const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const WasmPackPlugin = require("@wasm-tool/wasm-pack-plugin");

module.exports = (_, argv) => {
    console.log('Building in %s mode', argv.mode);
    config = {
        entry: './index.ts',
        resolve: {
            extensions: ['.ts', '.js'],
        },
        module: {
            rules: [
                {
                    test: /\.ts$/,
                    use: 'ts-loader',
                    exclude: /node_modules/,
                },
            ],
        },
        output: {
            path: path.resolve(__dirname, 'dist'),
            filename: 'index.js',
        },
        plugins: [
            new HtmlWebpackPlugin({
                template: 'index.html'
            }),
            new WasmPackPlugin({
                crateDirectory: path.resolve(__dirname, ".")
            })
        ],
        experiments: {
            asyncWebAssembly: true
        }
    };
    return config;
}