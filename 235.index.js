"use strict";(self.webpackChunk=self.webpackChunk||[]).push([[235],{235:(e,_,t)=>{t.a(e,(async(e,r)=>{try{t.r(_),t.d(_,{Simulation:()=>a.uL,__wbg_attachShader_cda29f0482c65440:()=>a._o,__wbg_bindBuffer_6f9a2fa9ebc65b01:()=>a.we,__wbg_bufferData_bc2f1a27f7162655:()=>a.Tw,__wbg_bufferSubData_dfaee0913e5a5aa9:()=>a.vt,__wbg_buffer_55ba7a6b1b92e2ac:()=>a.OQ,__wbg_clearColor_116005afbb6df00f:()=>a.FG,__wbg_clear_15c6565459b2686d:()=>a.y1,__wbg_compileShader_6f505d659e2795e6:()=>a.zt,__wbg_createBuffer_0da7eb27184081a8:()=>a.M7,__wbg_createProgram_535e1a7a84baa7ff:()=>a.bg,__wbg_createShader_b1a69c91a9abbcf9:()=>a.iu,__wbg_drawArrays_77814548b9e573f2:()=>a.T,__wbg_enableVertexAttribArray_7e45a67bd47ec1bc:()=>a.Ee,__wbg_getAttribLocation_9fb8d1fcf1e79c7d:()=>a.ds,__wbg_getProgramInfoLog_03d7941c48fa9179:()=>a.bI,__wbg_getProgramParameter_dd171792e4ba3184:()=>a.Nw,__wbg_getShaderInfoLog_c1cca646bf94aa17:()=>a.Pk,__wbg_getShaderParameter_c1d89b570b67be37:()=>a.dm,__wbg_getUniformLocation_984bcb57f0539335:()=>a.BF,__wbg_linkProgram_0a25df9d4086c8c9:()=>a.lk,__wbg_newwithbyteoffsetandlength_ab5b524f83702d8d:()=>a.qp,__wbg_set_wasm:()=>a.oT,__wbg_shaderSource_5c55ce208ee2dc38:()=>a.Q6,__wbg_uniform1i_ef0ff3d67b59f4de:()=>a.KP,__wbg_uniformMatrix4fv_43b24d28b294355e:()=>a.tX,__wbg_useProgram_f16b06e2ecdf168f:()=>a.ls,__wbg_vertexAttribPointer_c16280a7c840a534:()=>a.EF,__wbg_viewport_a79678835091995e:()=>a.gd,__wbindgen_boolean_get:()=>a.HT,__wbindgen_memory:()=>a.oH,__wbindgen_object_drop_ref:()=>a.ug,__wbindgen_string_new:()=>a.h4,__wbindgen_throw:()=>a.Or});var n=t(530),a=t(838),o=e([n]);n=(o.then?(await o)():o)[0],(0,a.oT)(n),r()}catch(e){r(e)}}))},838:(e,_,t)=>{let r;function n(e){r=e}t.d(_,{BF:()=>q,EF:()=>N,Ee:()=>V,FG:()=>B,HT:()=>T,KP:()=>U,M7:()=>G,Nw:()=>C,OQ:()=>J,Or:()=>ee,Pk:()=>I,Q6:()=>x,T:()=>Z,Tw:()=>W,_o:()=>O,bI:()=>H,bg:()=>D,dm:()=>k,ds:()=>z,gd:()=>F,h4:()=>E,iu:()=>S,lk:()=>M,ls:()=>Q,oH:()=>X,oT:()=>n,qp:()=>R,tX:()=>j,uL:()=>v,ug:()=>L,vt:()=>$,we:()=>K,y1:()=>Y,zt:()=>A}),e=t.hmd(e);const a=new Array(128).fill(void 0);function o(e){return a[e]}a.push(void 0,null,!0,!1);let b=a.length;function c(e){const _=o(e);return function(e){e<132||(a[e]=b,b=e)}(e),_}let f=new("undefined"==typeof TextDecoder?(0,e.require)("util").TextDecoder:TextDecoder)("utf-8",{ignoreBOM:!0,fatal:!0});f.decode();let i=null;function g(){return null!==i&&0!==i.byteLength||(i=new Uint8Array(r.memory.buffer)),i}function u(e,_){return e>>>=0,f.decode(g().subarray(e,e+_))}function d(e){b===a.length&&a.push(a.length+1);const _=b;return b=a[_],a[_]=e,_}let w=null;function s(){return null!==w&&0!==w.byteLength||(w=new Int32Array(r.memory.buffer)),w}let l=null;function h(e){return null==e}let m=0,p=new("undefined"==typeof TextEncoder?(0,e.require)("util").TextEncoder:TextEncoder)("utf-8");const y="function"==typeof p.encodeInto?function(e,_){return p.encodeInto(e,_)}:function(e,_){const t=p.encode(e);return _.set(t),{read:e.length,written:t.length}};function P(e,_,t){if(void 0===t){const t=p.encode(e),r=_(t.length)>>>0;return g().subarray(r,r+t.length).set(t),m=t.length,r}let r=e.length,n=_(r)>>>0;const a=g();let o=0;for(;o<r;o++){const _=e.charCodeAt(o);if(_>127)break;a[n+o]=_}if(o!==r){0!==o&&(e=e.slice(o)),n=t(n,r,r=o+3*e.length)>>>0;const _=g().subarray(n+o,n+r);o+=y(e,_).written}return m=o,n}class v{static __wrap(e){e>>>=0;const _=Object.create(v.prototype);return _.__wbg_ptr=e,_}__destroy_into_raw(){const e=this.__wbg_ptr;return this.__wbg_ptr=0,e}free(){const e=this.__destroy_into_raw();r.__wbg_simulation_free(e)}constructor(e,_,t,n){try{const b=r.__wbindgen_add_to_stack_pointer(-16);r.simulation_new(b,d(e),_,t,n);var a=s()[b/4+0],o=s()[b/4+1];if(s()[b/4+2])throw c(o);return v.__wrap(a)}finally{r.__wbindgen_add_to_stack_pointer(16)}}set draw_single_color(e){r.simulation_set_draw_single_color(this.__wbg_ptr,e)}get num_particles(){return r.simulation_num_particles(this.__wbg_ptr)>>>0}set viscosity(e){r.simulation_set_viscosity(this.__wbg_ptr,e)}set solver_substeps(e){r.simulation_set_solver_substeps(this.__wbg_ptr,e)}step(){r.simulation_step(this.__wbg_ptr)}add_block(){r.simulation_add_block(this.__wbg_ptr)}reset(e,_){r.simulation_reset(this.__wbg_ptr,e,_)}draw(){r.simulation_draw(this.__wbg_ptr)}}function S(e,_){const t=o(e).createShader(_>>>0);return h(t)?0:d(t)}function x(e,_,t,r){o(e).shaderSource(o(_),u(t,r))}function A(e,_){o(e).compileShader(o(_))}function k(e,_,t){return d(o(e).getShaderParameter(o(_),t>>>0))}function T(e){const _=o(e);return"boolean"==typeof _?_?1:0:2}function L(e){c(e)}function I(e,_,t){const n=o(_).getShaderInfoLog(o(t));var a=h(n)?0:P(n,r.__wbindgen_export_0,r.__wbindgen_export_1),b=m;s()[e/4+1]=b,s()[e/4+0]=a}function F(e,_,t,r,n){o(e).viewport(_,t,r,n)}function B(e,_,t,r,n){o(e).clearColor(_,t,r,n)}function E(e,_){return d(u(e,_))}function D(e){const _=o(e).createProgram();return h(_)?0:d(_)}function O(e,_,t){o(e).attachShader(o(_),o(t))}function M(e,_){o(e).linkProgram(o(_))}function C(e,_,t){return d(o(e).getProgramParameter(o(_),t>>>0))}function H(e,_,t){const n=o(_).getProgramInfoLog(o(t));var a=h(n)?0:P(n,r.__wbindgen_export_0,r.__wbindgen_export_1),b=m;s()[e/4+1]=b,s()[e/4+0]=a}function Q(e,_){o(e).useProgram(o(_))}function q(e,_,t,r){const n=o(e).getUniformLocation(o(_),u(t,r));return h(n)?0:d(n)}function j(e,_,t,n,a){var b,c;o(e).uniformMatrix4fv(o(_),0!==t,(b=n,c=a,b>>>=0,(null!==l&&0!==l.byteLength||(l=new Float32Array(r.memory.buffer)),l).subarray(b/4,b/4+c)))}function U(e,_,t){o(e).uniform1i(o(_),t)}function z(e,_,t,r){return o(e).getAttribLocation(o(_),u(t,r))}function G(e){const _=o(e).createBuffer();return h(_)?0:d(_)}function K(e,_,t){o(e).bindBuffer(_>>>0,o(t))}function N(e,_,t,r,n,a,b){o(e).vertexAttribPointer(_>>>0,t,r>>>0,0!==n,a,b)}function V(e,_){o(e).enableVertexAttribArray(_>>>0)}function X(){return d(r.memory)}function J(e){return d(o(e).buffer)}function R(e,_,t){return d(new Float32Array(o(e),_>>>0,t>>>0))}function W(e,_,t,r){o(e).bufferData(_>>>0,o(t),r>>>0)}function Y(e,_){o(e).clear(_>>>0)}function Z(e,_,t,r){o(e).drawArrays(_>>>0,t,r)}function $(e,_,t,r){o(e).bufferSubData(_>>>0,t,o(r))}function ee(e,_){throw new Error(u(e,_))}},530:(e,_,t)=>{var r=t(838);e.exports=t.v(_,e.id,"cc91560424df7a2c85ee",{"./index_bg.js":{__wbg_createShader_b1a69c91a9abbcf9:r.iu,__wbg_shaderSource_5c55ce208ee2dc38:r.Q6,__wbg_compileShader_6f505d659e2795e6:r.zt,__wbg_getShaderParameter_c1d89b570b67be37:r.dm,__wbindgen_boolean_get:r.HT,__wbindgen_object_drop_ref:r.ug,__wbg_getShaderInfoLog_c1cca646bf94aa17:r.Pk,__wbg_viewport_a79678835091995e:r.gd,__wbg_clearColor_116005afbb6df00f:r.FG,__wbindgen_string_new:r.h4,__wbg_createProgram_535e1a7a84baa7ff:r.bg,__wbg_attachShader_cda29f0482c65440:r._o,__wbg_linkProgram_0a25df9d4086c8c9:r.lk,__wbg_getProgramParameter_dd171792e4ba3184:r.Nw,__wbg_getProgramInfoLog_03d7941c48fa9179:r.bI,__wbg_useProgram_f16b06e2ecdf168f:r.ls,__wbg_getUniformLocation_984bcb57f0539335:r.BF,__wbg_uniformMatrix4fv_43b24d28b294355e:r.tX,__wbg_uniform1i_ef0ff3d67b59f4de:r.KP,__wbg_getAttribLocation_9fb8d1fcf1e79c7d:r.ds,__wbg_createBuffer_0da7eb27184081a8:r.M7,__wbg_bindBuffer_6f9a2fa9ebc65b01:r.we,__wbg_vertexAttribPointer_c16280a7c840a534:r.EF,__wbg_enableVertexAttribArray_7e45a67bd47ec1bc:r.Ee,__wbindgen_memory:r.oH,__wbg_buffer_55ba7a6b1b92e2ac:r.OQ,__wbg_newwithbyteoffsetandlength_ab5b524f83702d8d:r.qp,__wbg_bufferData_bc2f1a27f7162655:r.Tw,__wbg_clear_15c6565459b2686d:r.y1,__wbg_drawArrays_77814548b9e573f2:r.T,__wbg_bufferSubData_dfaee0913e5a5aa9:r.vt,__wbindgen_throw:r.Or}})}}]);