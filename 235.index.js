"use strict";(self.webpackChunk=self.webpackChunk||[]).push([[235],{235:(e,t,_)=>{_.a(e,(async(e,r)=>{try{_.r(t),_.d(t,{Simulation:()=>n.uL,__wbg_attachShader_55dbe770f3ee32ca:()=>n.y1,__wbg_bindBuffer_29d52e7bc48650c3:()=>n.ro,__wbg_bufferData_85d635f32a990208:()=>n.$V,__wbg_bufferSubData_3a944e1fdad0cd9a:()=>n.I5,__wbg_buffer_5e74a88a1424a2e0:()=>n.rf,__wbg_clearColor_51c4f69c743c3252:()=>n.jN,__wbg_clear_2af1271959ec83d7:()=>n.Dr,__wbg_compileShader_3b5f9ef4c67a0777:()=>n.Ih,__wbg_createBuffer_c40f37e1348bb91f:()=>n.AF,__wbg_createProgram_245520da1fb9e47b:()=>n.E8,__wbg_createShader_4d8818a13cb825b3:()=>n.wn,__wbg_drawArrays_22c88d644a33fd59:()=>n.XR,__wbg_enableVertexAttribArray_4ed5f91d0718bee1:()=>n.Rh,__wbg_getAttribLocation_da5df7094096113d:()=>n.Ux,__wbg_getContext_bd4e9445094eda84:()=>n.c7,__wbg_getProgramInfoLog_c253042b64e86027:()=>n.Wo,__wbg_getProgramParameter_4f698af0dda0a2d4:()=>n.Dd,__wbg_getShaderInfoLog_584794e3bcf1e19b:()=>n.HM,__wbg_getShaderParameter_64b1ffe576e5fa25:()=>n.dV,__wbg_getUniformLocation_703972f150a46500:()=>n.hN,__wbg_instanceof_WebGlRenderingContext_2be4c068bf5f8362:()=>n.I7,__wbg_linkProgram_5fdd57237c761833:()=>n.Xb,__wbg_newwithbyteoffsetandlength_ad2916c6fa7d4c6f:()=>n.HK,__wbg_setheight_28f53831182cc410:()=>n.Si,__wbg_setwidth_362e8db8cbadbe96:()=>n.RA,__wbg_shaderSource_173ab97288934a60:()=>n.BI,__wbg_uniform4fv_481536ab64fdd3a3:()=>n.Dt,__wbg_uniformMatrix4fv_f07c6caf5a563616:()=>n.zw,__wbg_useProgram_d5898a40ebe88916:()=>n.lV,__wbg_vertexAttribPointer_0d097efa33e3f45f:()=>n.sb,__wbg_viewport_19577064127daf83:()=>n.N_,__wbindgen_boolean_get:()=>n.HT,__wbindgen_memory:()=>n.oH,__wbindgen_object_drop_ref:()=>n.ug,__wbindgen_string_new:()=>n.h4,__wbindgen_throw:()=>n.Or});var n=_(838),a=e([n]);n=(a.then?(await a)():a)[0],r()}catch(e){r(e)}}))},838:(e,t,_)=>{_.a(e,(async(r,n)=>{try{_.d(t,{uL:()=>k,wn:()=>C,BI:()=>H,Ih:()=>R,dV:()=>V,HT:()=>B,ug:()=>N,HM:()=>T,RA:()=>j,Si:()=>E,c7:()=>M,I7:()=>U,N_:()=>W,jN:()=>X,h4:()=>F,E8:()=>O,y1:()=>z,Xb:()=>G,Dd:()=>K,Wo:()=>$,lV:()=>q,hN:()=>J,zw:()=>Q,Ux:()=>Y,AF:()=>Z,ro:()=>ee,sb:()=>te,Rh:()=>_e,oH:()=>re,rf:()=>ne,HK:()=>ae,$V:()=>oe,Dr:()=>fe,Dt:()=>ie,XR:()=>ce,I5:()=>be,Or:()=>ue});var a=_(530);e=_.hmd(e);var o=r([a]);a=(o.then?(await o)():o)[0];const f=new Array(32).fill(void 0);function i(e){return f[e]}f.push(void 0,null,!0,!1);let c=f.length;function b(e){e<36||(f[e]=c,c=e)}function u(e){const t=i(e);return b(e),t}let d=new("undefined"==typeof TextDecoder?(0,e.require)("util").TextDecoder:TextDecoder)("utf-8",{ignoreBOM:!0,fatal:!0});d.decode();let g=null;function w(){return null!==g&&g.buffer===a.memory.buffer||(g=new Uint8Array(a.memory.buffer)),g}function s(e,t){return d.decode(w().subarray(e,e+t))}function l(e){c===f.length&&f.push(f.length+1);const t=c;return c=f[t],f[t]=e,t}let h=32;function m(e){if(1==h)throw new Error("out of js stack");return f[--h]=e,h}let y=null;function p(){return null!==y&&y.buffer===a.memory.buffer||(y=new Int32Array(a.memory.buffer)),y}function v(e){return null==e}function A(e,t){try{return e.apply(this,t)}catch(e){a.__wbindgen_exn_store(l(e))}}let x=null;function S(e,t){return(null!==x&&x.buffer===a.memory.buffer||(x=new Float32Array(a.memory.buffer)),x).subarray(e/4,e/4+t)}let P=0,I=new("undefined"==typeof TextEncoder?(0,e.require)("util").TextEncoder:TextEncoder)("utf-8");const D="function"==typeof I.encodeInto?function(e,t){return I.encodeInto(e,t)}:function(e,t){const _=I.encode(e);return t.set(_),{read:e.length,written:_.length}};function L(e,t,_){if(void 0===_){const _=I.encode(e),r=t(_.length);return w().subarray(r,r+_.length).set(_),P=_.length,r}let r=e.length,n=t(r);const a=w();let o=0;for(;o<r;o++){const t=e.charCodeAt(o);if(t>127)break;a[n+o]=t}if(o!==r){0!==o&&(e=e.slice(o)),n=_(n,r,r=o+3*e.length);const t=w().subarray(n+o,n+r);o+=D(e,t).written}return P=o,n}class k{static __wrap(e){const t=Object.create(k.prototype);return t.ptr=e,t}__destroy_into_raw(){const e=this.ptr;return this.ptr=0,e}free(){const e=this.__destroy_into_raw();a.__wbg_simulation_free(e)}constructor(e){try{const r=a.__wbindgen_add_to_stack_pointer(-16);a.simulation_new(r,m(e));var t=p()[r/4+0],_=p()[r/4+1];if(p()[r/4+2])throw u(_);return k.__wrap(t)}finally{a.__wbindgen_add_to_stack_pointer(16),f[h++]=void 0}}get_num_particles(){return a.simulation_get_num_particles(this.ptr)>>>0}set_viscosity(e){a.simulation_set_viscosity(this.ptr,e)}set_solver_substeps(e){a.simulation_set_solver_substeps(this.ptr,e)}step(){a.simulation_step(this.ptr)}add_block(){a.simulation_add_block(this.ptr)}reset(){a.simulation_reset(this.ptr)}}function C(e,t){var _=i(e).createShader(t>>>0);return v(_)?0:l(_)}function H(e,t,_,r){i(e).shaderSource(i(t),s(_,r))}function R(e,t){i(e).compileShader(i(t))}function V(e,t,_){return l(i(e).getShaderParameter(i(t),_>>>0))}function B(e){const t=i(e);return"boolean"==typeof t?t?1:0:2}function N(e){u(e)}function T(e,t,_){var r=i(t).getShaderInfoLog(i(_)),n=v(r)?0:L(r,a.__wbindgen_malloc,a.__wbindgen_realloc),o=P;p()[e/4+1]=o,p()[e/4+0]=n}function j(e,t){i(e).width=t>>>0}function E(e,t){i(e).height=t>>>0}function M(){return A((function(e,t,_){var r=i(e).getContext(s(t,_));return v(r)?0:l(r)}),arguments)}function U(e){return i(e)instanceof WebGLRenderingContext}function W(e,t,_,r,n){i(e).viewport(t,_,r,n)}function X(e,t,_,r,n){i(e).clearColor(t,_,r,n)}function F(e,t){return l(s(e,t))}function O(e){var t=i(e).createProgram();return v(t)?0:l(t)}function z(e,t,_){i(e).attachShader(i(t),i(_))}function G(e,t){i(e).linkProgram(i(t))}function K(e,t,_){return l(i(e).getProgramParameter(i(t),_>>>0))}function $(e,t,_){var r=i(t).getProgramInfoLog(i(_)),n=v(r)?0:L(r,a.__wbindgen_malloc,a.__wbindgen_realloc),o=P;p()[e/4+1]=o,p()[e/4+0]=n}function q(e,t){i(e).useProgram(i(t))}function J(e,t,_,r){var n=i(e).getUniformLocation(i(t),s(_,r));return v(n)?0:l(n)}function Q(e,t,_,r,n){i(e).uniformMatrix4fv(i(t),0!==_,S(r,n))}function Y(e,t,_,r){return i(e).getAttribLocation(i(t),s(_,r))}function Z(e){var t=i(e).createBuffer();return v(t)?0:l(t)}function ee(e,t,_){i(e).bindBuffer(t>>>0,i(_))}function te(e,t,_,r,n,a,o){i(e).vertexAttribPointer(t>>>0,_,r>>>0,0!==n,a,o)}function _e(e,t){i(e).enableVertexAttribArray(t>>>0)}function re(){return l(a.memory)}function ne(e){return l(i(e).buffer)}function ae(e,t,_){return l(new Float32Array(i(e),t>>>0,_>>>0))}function oe(e,t,_,r){i(e).bufferData(t>>>0,i(_),r>>>0)}function fe(e,t){i(e).clear(t>>>0)}function ie(e,t,_,r){i(e).uniform4fv(i(t),S(_,r))}function ce(e,t,_,r){i(e).drawArrays(t>>>0,_,r)}function be(e,t,_,r){i(e).bufferSubData(t>>>0,_,i(r))}function ue(e,t){throw new Error(s(e,t))}n()}catch(e){n(e)}}))},530:(e,t,_)=>{_.a(e,(async(r,n)=>{try{var a,o=r([a=_(838)]),[a]=o.then?(await o)():o;await _.v(t,e.id,"f03971306f2683fb4b65",{"./index_bg.js":{__wbg_createShader_4d8818a13cb825b3:a.wn,__wbg_shaderSource_173ab97288934a60:a.BI,__wbg_compileShader_3b5f9ef4c67a0777:a.Ih,__wbg_getShaderParameter_64b1ffe576e5fa25:a.dV,__wbindgen_boolean_get:a.HT,__wbindgen_object_drop_ref:a.ug,__wbg_getShaderInfoLog_584794e3bcf1e19b:a.HM,__wbg_setwidth_362e8db8cbadbe96:a.RA,__wbg_setheight_28f53831182cc410:a.Si,__wbg_getContext_bd4e9445094eda84:a.c7,__wbg_instanceof_WebGlRenderingContext_2be4c068bf5f8362:a.I7,__wbg_viewport_19577064127daf83:a.N_,__wbg_clearColor_51c4f69c743c3252:a.jN,__wbindgen_string_new:a.h4,__wbg_createProgram_245520da1fb9e47b:a.E8,__wbg_attachShader_55dbe770f3ee32ca:a.y1,__wbg_linkProgram_5fdd57237c761833:a.Xb,__wbg_getProgramParameter_4f698af0dda0a2d4:a.Dd,__wbg_getProgramInfoLog_c253042b64e86027:a.Wo,__wbg_useProgram_d5898a40ebe88916:a.lV,__wbg_getUniformLocation_703972f150a46500:a.hN,__wbg_uniformMatrix4fv_f07c6caf5a563616:a.zw,__wbg_getAttribLocation_da5df7094096113d:a.Ux,__wbg_createBuffer_c40f37e1348bb91f:a.AF,__wbg_bindBuffer_29d52e7bc48650c3:a.ro,__wbg_vertexAttribPointer_0d097efa33e3f45f:a.sb,__wbg_enableVertexAttribArray_4ed5f91d0718bee1:a.Rh,__wbindgen_memory:a.oH,__wbg_buffer_5e74a88a1424a2e0:a.rf,__wbg_newwithbyteoffsetandlength_ad2916c6fa7d4c6f:a.HK,__wbg_bufferData_85d635f32a990208:a.$V,__wbg_clear_2af1271959ec83d7:a.Dr,__wbg_uniform4fv_481536ab64fdd3a3:a.Dt,__wbg_drawArrays_22c88d644a33fd59:a.XR,__wbg_bufferSubData_3a944e1fdad0cd9a:a.I5,__wbindgen_throw:a.Or}}),n()}catch(e){n(e)}}),1)}}]);