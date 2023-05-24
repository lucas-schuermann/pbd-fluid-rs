/*! For license information please see index.js.LICENSE.txt */
(()=>{var t,e,i,n,s,r={466:function(t){var e;t.exports=((e=function(){function t(t){return s.appendChild(t.dom),t}function i(t){for(var e=0;e<s.children.length;e++)s.children[e].style.display=e===t?"block":"none";n=t}var n=0,s=document.createElement("div");s.style.cssText="position:fixed;top:0;left:0;cursor:pointer;opacity:0.9;z-index:10000",s.addEventListener("click",(function(t){t.preventDefault(),i(++n%s.children.length)}),!1);var r=(performance||Date).now(),l=r,o=0,a=t(new e.Panel("FPS","#0ff","#002")),h=t(new e.Panel("MS","#0f0","#020"));if(self.performance&&self.performance.memory)var d=t(new e.Panel("MB","#f08","#201"));return i(0),{REVISION:16,dom:s,addPanel:t,showPanel:i,begin:function(){r=(performance||Date).now()},end:function(){o++;var t=(performance||Date).now();if(h.update(t-r,200),t>l+1e3&&(a.update(1e3*o/(t-l),100),l=t,o=0,d)){var e=performance.memory;d.update(e.usedJSHeapSize/1048576,e.jsHeapSizeLimit/1048576)}return t},update:function(){r=this.end()},domElement:s,setMode:i}}).Panel=function(t,e,i){var n=1/0,s=0,r=Math.round,l=r(window.devicePixelRatio||1),o=80*l,a=48*l,h=3*l,d=2*l,c=3*l,u=15*l,p=74*l,g=30*l,m=document.createElement("canvas");m.width=o,m.height=a,m.style.cssText="width:80px;height:48px";var b=m.getContext("2d");return b.font="bold "+9*l+"px Helvetica,Arial,sans-serif",b.textBaseline="top",b.fillStyle=i,b.fillRect(0,0,o,a),b.fillStyle=e,b.fillText(t,h,d),b.fillRect(c,u,p,g),b.fillStyle=i,b.globalAlpha=.9,b.fillRect(c,u,p,g),{dom:m,update:function(a,v){n=Math.min(n,a),s=Math.max(s,a),b.fillStyle=i,b.globalAlpha=1,b.fillRect(0,0,o,u),b.fillStyle=e,b.fillText(r(a)+" "+t+" ("+r(n)+"-"+r(s)+")",h,d),b.drawImage(m,c+l,u,p-l,g,c,u,p-l,g),b.fillRect(c+p-l,u,l,g),b.fillStyle=i,b.globalAlpha=.9,b.fillRect(c+p-l,u,l,r((1-a/v)*g))}}},e)}},l={};function o(t){var e=l[t];if(void 0!==e)return e.exports;var i=l[t]={id:t,loaded:!1,exports:{}};return r[t].call(i.exports,i,i.exports,o),i.loaded=!0,i.exports}o.m=r,t="function"==typeof Symbol?Symbol("webpack queues"):"__webpack_queues__",e="function"==typeof Symbol?Symbol("webpack exports"):"__webpack_exports__",i="function"==typeof Symbol?Symbol("webpack error"):"__webpack_error__",n=t=>{t&&!t.d&&(t.d=1,t.forEach((t=>t.r--)),t.forEach((t=>t.r--?t.r++:t())))},o.a=(s,r,l)=>{var o;l&&((o=[]).d=1);var a,h,d,c=new Set,u=s.exports,p=new Promise(((t,e)=>{d=e,h=t}));p[e]=u,p[t]=t=>(o&&t(o),c.forEach(t),p.catch((t=>{}))),s.exports=p,r((s=>{var r;a=(s=>s.map((s=>{if(null!==s&&"object"==typeof s){if(s[t])return s;if(s.then){var r=[];r.d=0,s.then((t=>{l[e]=t,n(r)}),(t=>{l[i]=t,n(r)}));var l={};return l[t]=t=>t(r),l}}var o={};return o[t]=t=>{},o[e]=s,o})))(s);var l=()=>a.map((t=>{if(t[i])throw t[i];return t[e]})),h=new Promise((e=>{(r=()=>e(l)).r=0;var i=t=>t!==o&&!c.has(t)&&(c.add(t),t&&!t.d&&(r.r++,t.push(r)));a.map((e=>e[t](i)))}));return r.r?h:l()}),(t=>(t?d(p[i]=t):h(u),n(o)))),o&&(o.d=0)},o.d=(t,e)=>{for(var i in e)o.o(e,i)&&!o.o(t,i)&&Object.defineProperty(t,i,{enumerable:!0,get:e[i]})},o.f={},o.e=t=>Promise.all(Object.keys(o.f).reduce(((e,i)=>(o.f[i](t,e),e)),[])),o.u=t=>t+".index.js",o.g=function(){if("object"==typeof globalThis)return globalThis;try{return this||new Function("return this")()}catch(t){if("object"==typeof window)return window}}(),o.hmd=t=>((t=Object.create(t)).children||(t.children=[]),Object.defineProperty(t,"exports",{enumerable:!0,set:()=>{throw new Error("ES Modules may not assign module.exports or exports.*, Use ESM export syntax, instead: "+t.id)}}),t),o.o=(t,e)=>Object.prototype.hasOwnProperty.call(t,e),s={},o.l=(t,e,i,n)=>{if(s[t])s[t].push(e);else{var r,l;if(void 0!==i)for(var a=document.getElementsByTagName("script"),h=0;h<a.length;h++){var d=a[h];if(d.getAttribute("src")==t){r=d;break}}r||(l=!0,(r=document.createElement("script")).charset="utf-8",r.timeout=120,o.nc&&r.setAttribute("nonce",o.nc),r.src=t),s[t]=[e];var c=(e,i)=>{r.onerror=r.onload=null,clearTimeout(u);var n=s[t];if(delete s[t],r.parentNode&&r.parentNode.removeChild(r),n&&n.forEach((t=>t(i))),e)return e(i)},u=setTimeout(c.bind(null,void 0,{type:"timeout",target:r}),12e4);r.onerror=c.bind(null,r.onerror),r.onload=c.bind(null,r.onload),l&&document.head.appendChild(r)}},o.r=t=>{"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(t,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(t,"__esModule",{value:!0})},o.v=(t,e,i,n)=>{var s=fetch(o.p+""+i+".module.wasm");return"function"==typeof WebAssembly.instantiateStreaming?WebAssembly.instantiateStreaming(s,n).then((e=>Object.assign(t,e.instance.exports))):s.then((t=>t.arrayBuffer())).then((t=>WebAssembly.instantiate(t,n))).then((e=>Object.assign(t,e.instance.exports)))},(()=>{var t;o.g.importScripts&&(t=o.g.location+"");var e=o.g.document;if(!t&&e&&(e.currentScript&&(t=e.currentScript.src),!t)){var i=e.getElementsByTagName("script");if(i.length)for(var n=i.length-1;n>-1&&!t;)t=i[n--].src}if(!t)throw new Error("Automatic publicPath is not supported in this browser");t=t.replace(/#.*$/,"").replace(/\?.*$/,"").replace(/\/[^\/]+$/,"/"),o.p=t})(),(()=>{var t={179:0};o.f.j=(e,i)=>{var n=o.o(t,e)?t[e]:void 0;if(0!==n)if(n)i.push(n[2]);else{var s=new Promise(((i,s)=>n=t[e]=[i,s]));i.push(n[2]=s);var r=o.p+o.u(e),l=new Error;o.l(r,(i=>{if(o.o(t,e)&&(0!==(n=t[e])&&(t[e]=void 0),n)){var s=i&&("load"===i.type?"missing":i.type),r=i&&i.target&&i.target.src;l.message="Loading chunk "+e+" failed.\n("+s+": "+r+")",l.name="ChunkLoadError",l.type=s,l.request=r,n[1](l)}}),"chunk-"+e,e)}};var e=(e,i)=>{var n,s,[r,l,a]=i,h=0;if(r.some((e=>0!==t[e]))){for(n in l)o.o(l,n)&&(o.m[n]=l[n]);a&&a(o)}for(e&&e(i);h<r.length;h++)s=r[h],o.o(t,s)&&t[s]&&t[s][0](),t[s]=0},i=self.webpackChunk=self.webpackChunk||[];i.forEach(e.bind(null,0)),i.push=e.bind(null,i.push.bind(i))})(),(()=>{"use strict";var t=o(466);class e{constructor(t,i,n,s,r="div"){this.parent=t,this.object=i,this.property=n,this._disabled=!1,this._hidden=!1,this.initialValue=this.getValue(),this.domElement=document.createElement("div"),this.domElement.classList.add("controller"),this.domElement.classList.add(s),this.$name=document.createElement("div"),this.$name.classList.add("name"),e.nextNameID=e.nextNameID||0,this.$name.id="lil-gui-name-"+ ++e.nextNameID,this.$widget=document.createElement(r),this.$widget.classList.add("widget"),this.$disable=this.$widget,this.domElement.appendChild(this.$name),this.domElement.appendChild(this.$widget),this.parent.children.push(this),this.parent.controllers.push(this),this.parent.$children.appendChild(this.domElement),this._listenCallback=this._listenCallback.bind(this),this.name(n)}name(t){return this._name=t,this.$name.innerHTML=t,this}onChange(t){return this._onChange=t,this}_callOnChange(){this.parent._callOnChange(this),void 0!==this._onChange&&this._onChange.call(this,this.getValue()),this._changed=!0}onFinishChange(t){return this._onFinishChange=t,this}_callOnFinishChange(){this._changed&&(this.parent._callOnFinishChange(this),void 0!==this._onFinishChange&&this._onFinishChange.call(this,this.getValue())),this._changed=!1}reset(){return this.setValue(this.initialValue),this._callOnFinishChange(),this}enable(t=!0){return this.disable(!t)}disable(t=!0){return t===this._disabled||(this._disabled=t,this.domElement.classList.toggle("disabled",t),this.$disable.toggleAttribute("disabled",t)),this}show(t=!0){return this._hidden=!t,this.domElement.style.display=this._hidden?"none":"",this}hide(){return this.show(!1)}options(t){const e=this.parent.add(this.object,this.property,t);return e.name(this._name),this.destroy(),e}min(t){return this}max(t){return this}step(t){return this}decimals(t){return this}listen(t=!0){return this._listening=t,void 0!==this._listenCallbackID&&(cancelAnimationFrame(this._listenCallbackID),this._listenCallbackID=void 0),this._listening&&this._listenCallback(),this}_listenCallback(){this._listenCallbackID=requestAnimationFrame(this._listenCallback);const t=this.save();t!==this._listenPrevValue&&this.updateDisplay(),this._listenPrevValue=t}getValue(){return this.object[this.property]}setValue(t){return this.object[this.property]=t,this._callOnChange(),this.updateDisplay(),this}updateDisplay(){return this}load(t){return this.setValue(t),this._callOnFinishChange(),this}save(){return this.getValue()}destroy(){this.listen(!1),this.parent.children.splice(this.parent.children.indexOf(this),1),this.parent.controllers.splice(this.parent.controllers.indexOf(this),1),this.parent.$children.removeChild(this.domElement)}}class i extends e{constructor(t,e,i){super(t,e,i,"boolean","label"),this.$input=document.createElement("input"),this.$input.setAttribute("type","checkbox"),this.$input.setAttribute("aria-labelledby",this.$name.id),this.$widget.appendChild(this.$input),this.$input.addEventListener("change",(()=>{this.setValue(this.$input.checked),this._callOnFinishChange()})),this.$disable=this.$input,this.updateDisplay()}updateDisplay(){return this.$input.checked=this.getValue(),this}}function n(t){let e,i;return(e=t.match(/(#|0x)?([a-f0-9]{6})/i))?i=e[2]:(e=t.match(/rgb\(\s*(\d*)\s*,\s*(\d*)\s*,\s*(\d*)\s*\)/))?i=parseInt(e[1]).toString(16).padStart(2,0)+parseInt(e[2]).toString(16).padStart(2,0)+parseInt(e[3]).toString(16).padStart(2,0):(e=t.match(/^#?([a-f0-9])([a-f0-9])([a-f0-9])$/i))&&(i=e[1]+e[1]+e[2]+e[2]+e[3]+e[3]),!!i&&"#"+i}const s={isPrimitive:!0,match:t=>"number"==typeof t,fromHexString:t=>parseInt(t.substring(1),16),toHexString:t=>"#"+t.toString(16).padStart(6,0)},r={isPrimitive:!1,match:t=>Array.isArray(t),fromHexString(t,e,i=1){const n=s.fromHexString(t);e[0]=(n>>16&255)/255*i,e[1]=(n>>8&255)/255*i,e[2]=(255&n)/255*i},toHexString:([t,e,i],n=1)=>s.toHexString(t*(n=255/n)<<16^e*n<<8^i*n<<0)},l={isPrimitive:!1,match:t=>Object(t)===t,fromHexString(t,e,i=1){const n=s.fromHexString(t);e.r=(n>>16&255)/255*i,e.g=(n>>8&255)/255*i,e.b=(255&n)/255*i},toHexString:({r:t,g:e,b:i},n=1)=>s.toHexString(t*(n=255/n)<<16^e*n<<8^i*n<<0)},a=[{isPrimitive:!0,match:t=>"string"==typeof t,fromHexString:n,toHexString:n},s,r,l];class h extends e{constructor(t,e,i,s){var r;super(t,e,i,"color"),this.$input=document.createElement("input"),this.$input.setAttribute("type","color"),this.$input.setAttribute("tabindex",-1),this.$input.setAttribute("aria-labelledby",this.$name.id),this.$text=document.createElement("input"),this.$text.setAttribute("type","text"),this.$text.setAttribute("spellcheck","false"),this.$text.setAttribute("aria-labelledby",this.$name.id),this.$display=document.createElement("div"),this.$display.classList.add("display"),this.$display.appendChild(this.$input),this.$widget.appendChild(this.$display),this.$widget.appendChild(this.$text),this._format=(r=this.initialValue,a.find((t=>t.match(r)))),this._rgbScale=s,this._initialValueHexString=this.save(),this._textFocused=!1,this.$input.addEventListener("input",(()=>{this._setValueFromHexString(this.$input.value)})),this.$input.addEventListener("blur",(()=>{this._callOnFinishChange()})),this.$text.addEventListener("input",(()=>{const t=n(this.$text.value);t&&this._setValueFromHexString(t)})),this.$text.addEventListener("focus",(()=>{this._textFocused=!0,this.$text.select()})),this.$text.addEventListener("blur",(()=>{this._textFocused=!1,this.updateDisplay(),this._callOnFinishChange()})),this.$disable=this.$text,this.updateDisplay()}reset(){return this._setValueFromHexString(this._initialValueHexString),this}_setValueFromHexString(t){if(this._format.isPrimitive){const e=this._format.fromHexString(t);this.setValue(e)}else this._format.fromHexString(t,this.getValue(),this._rgbScale),this._callOnChange(),this.updateDisplay()}save(){return this._format.toHexString(this.getValue(),this._rgbScale)}load(t){return this._setValueFromHexString(t),this._callOnFinishChange(),this}updateDisplay(){return this.$input.value=this._format.toHexString(this.getValue(),this._rgbScale),this._textFocused||(this.$text.value=this.$input.value.substring(1)),this.$display.style.backgroundColor=this.$input.value,this}}class d extends e{constructor(t,e,i){super(t,e,i,"function"),this.$button=document.createElement("button"),this.$button.appendChild(this.$name),this.$widget.appendChild(this.$button),this.$button.addEventListener("click",(t=>{t.preventDefault(),this.getValue().call(this.object),this._callOnChange()})),this.$button.addEventListener("touchstart",(()=>{}),{passive:!0}),this.$disable=this.$button}}class c extends e{constructor(t,e,i,n,s,r){super(t,e,i,"number"),this._initInput(),this.min(n),this.max(s);const l=void 0!==r;this.step(l?r:this._getImplicitStep(),l),this.updateDisplay()}decimals(t){return this._decimals=t,this.updateDisplay(),this}min(t){return this._min=t,this._onUpdateMinMax(),this}max(t){return this._max=t,this._onUpdateMinMax(),this}step(t,e=!0){return this._step=t,this._stepExplicit=e,this}updateDisplay(){const t=this.getValue();if(this._hasSlider){let e=(t-this._min)/(this._max-this._min);e=Math.max(0,Math.min(e,1)),this.$fill.style.width=100*e+"%"}return this._inputFocused||(this.$input.value=void 0===this._decimals?t:t.toFixed(this._decimals)),this}_initInput(){this.$input=document.createElement("input"),this.$input.setAttribute("type","number"),this.$input.setAttribute("step","any"),this.$input.setAttribute("aria-labelledby",this.$name.id),this.$widget.appendChild(this.$input),this.$disable=this.$input;const t=t=>{const e=parseFloat(this.$input.value);isNaN(e)||(this._snapClampSetValue(e+t),this.$input.value=this.getValue())};let e,i,n,s,r,l=!1;const o=t=>{if(l){const n=t.clientX-e,s=t.clientY-i;Math.abs(s)>5?(t.preventDefault(),this.$input.blur(),l=!1,this._setDraggingStyle(!0,"vertical")):Math.abs(n)>5&&a()}if(!l){const e=t.clientY-n;r-=e*this._step*this._arrowKeyMultiplier(t),s+r>this._max?r=this._max-s:s+r<this._min&&(r=this._min-s),this._snapClampSetValue(s+r)}n=t.clientY},a=()=>{this._setDraggingStyle(!1,"vertical"),this._callOnFinishChange(),window.removeEventListener("mousemove",o),window.removeEventListener("mouseup",a)};this.$input.addEventListener("input",(()=>{let t=parseFloat(this.$input.value);isNaN(t)||(this._stepExplicit&&(t=this._snap(t)),this.setValue(this._clamp(t)))})),this.$input.addEventListener("keydown",(e=>{"Enter"===e.code&&this.$input.blur(),"ArrowUp"===e.code&&(e.preventDefault(),t(this._step*this._arrowKeyMultiplier(e))),"ArrowDown"===e.code&&(e.preventDefault(),t(this._step*this._arrowKeyMultiplier(e)*-1))})),this.$input.addEventListener("wheel",(e=>{this._inputFocused&&(e.preventDefault(),t(this._step*this._normalizeMouseWheel(e)))}),{passive:!1}),this.$input.addEventListener("mousedown",(t=>{e=t.clientX,i=n=t.clientY,l=!0,s=this.getValue(),r=0,window.addEventListener("mousemove",o),window.addEventListener("mouseup",a)})),this.$input.addEventListener("focus",(()=>{this._inputFocused=!0})),this.$input.addEventListener("blur",(()=>{this._inputFocused=!1,this.updateDisplay(),this._callOnFinishChange()}))}_initSlider(){this._hasSlider=!0,this.$slider=document.createElement("div"),this.$slider.classList.add("slider"),this.$fill=document.createElement("div"),this.$fill.classList.add("fill"),this.$slider.appendChild(this.$fill),this.$widget.insertBefore(this.$slider,this.$input),this.domElement.classList.add("hasSlider");const t=t=>{const e=this.$slider.getBoundingClientRect();let i=(n=t,s=e.left,r=e.right,l=this._min,(n-s)/(r-s)*(this._max-l)+l);var n,s,r,l;this._snapClampSetValue(i)},e=e=>{t(e.clientX)},i=()=>{this._callOnFinishChange(),this._setDraggingStyle(!1),window.removeEventListener("mousemove",e),window.removeEventListener("mouseup",i)};let n,s,r=!1;const l=e=>{e.preventDefault(),this._setDraggingStyle(!0),t(e.touches[0].clientX),r=!1},o=e=>{if(r){const t=e.touches[0].clientX-n,i=e.touches[0].clientY-s;Math.abs(t)>Math.abs(i)?l(e):(window.removeEventListener("touchmove",o),window.removeEventListener("touchend",a))}else e.preventDefault(),t(e.touches[0].clientX)},a=()=>{this._callOnFinishChange(),this._setDraggingStyle(!1),window.removeEventListener("touchmove",o),window.removeEventListener("touchend",a)},h=this._callOnFinishChange.bind(this);let d;this.$slider.addEventListener("mousedown",(n=>{this._setDraggingStyle(!0),t(n.clientX),window.addEventListener("mousemove",e),window.addEventListener("mouseup",i)})),this.$slider.addEventListener("touchstart",(t=>{t.touches.length>1||(this._hasScrollBar?(n=t.touches[0].clientX,s=t.touches[0].clientY,r=!0):l(t),window.addEventListener("touchmove",o,{passive:!1}),window.addEventListener("touchend",a))}),{passive:!1}),this.$slider.addEventListener("wheel",(t=>{if(Math.abs(t.deltaX)<Math.abs(t.deltaY)&&this._hasScrollBar)return;t.preventDefault();const e=this._normalizeMouseWheel(t)*this._step;this._snapClampSetValue(this.getValue()+e),this.$input.value=this.getValue(),clearTimeout(d),d=setTimeout(h,400)}),{passive:!1})}_setDraggingStyle(t,e="horizontal"){this.$slider&&this.$slider.classList.toggle("active",t),document.body.classList.toggle("lil-gui-dragging",t),document.body.classList.toggle(`lil-gui-${e}`,t)}_getImplicitStep(){return this._hasMin&&this._hasMax?(this._max-this._min)/1e3:.1}_onUpdateMinMax(){!this._hasSlider&&this._hasMin&&this._hasMax&&(this._stepExplicit||this.step(this._getImplicitStep(),!1),this._initSlider(),this.updateDisplay())}_normalizeMouseWheel(t){let{deltaX:e,deltaY:i}=t;return Math.floor(t.deltaY)!==t.deltaY&&t.wheelDelta&&(e=0,i=-t.wheelDelta/120,i*=this._stepExplicit?1:10),e+-i}_arrowKeyMultiplier(t){let e=this._stepExplicit?1:10;return t.shiftKey?e*=10:t.altKey&&(e/=10),e}_snap(t){const e=Math.round(t/this._step)*this._step;return parseFloat(e.toPrecision(15))}_clamp(t){return t<this._min&&(t=this._min),t>this._max&&(t=this._max),t}_snapClampSetValue(t){this.setValue(this._clamp(this._snap(t)))}get _hasScrollBar(){const t=this.parent.root.$children;return t.scrollHeight>t.clientHeight}get _hasMin(){return void 0!==this._min}get _hasMax(){return void 0!==this._max}}class u extends e{constructor(t,e,i,n){super(t,e,i,"option"),this.$select=document.createElement("select"),this.$select.setAttribute("aria-labelledby",this.$name.id),this.$display=document.createElement("div"),this.$display.classList.add("display"),this._values=Array.isArray(n)?n:Object.values(n),this._names=Array.isArray(n)?n:Object.keys(n),this._names.forEach((t=>{const e=document.createElement("option");e.innerHTML=t,this.$select.appendChild(e)})),this.$select.addEventListener("change",(()=>{this.setValue(this._values[this.$select.selectedIndex]),this._callOnFinishChange()})),this.$select.addEventListener("focus",(()=>{this.$display.classList.add("focus")})),this.$select.addEventListener("blur",(()=>{this.$display.classList.remove("focus")})),this.$widget.appendChild(this.$select),this.$widget.appendChild(this.$display),this.$disable=this.$select,this.updateDisplay()}updateDisplay(){const t=this.getValue(),e=this._values.indexOf(t);return this.$select.selectedIndex=e,this.$display.innerHTML=-1===e?t:this._names[e],this}}class p extends e{constructor(t,e,i){super(t,e,i,"string"),this.$input=document.createElement("input"),this.$input.setAttribute("type","text"),this.$input.setAttribute("aria-labelledby",this.$name.id),this.$input.addEventListener("input",(()=>{this.setValue(this.$input.value)})),this.$input.addEventListener("keydown",(t=>{"Enter"===t.code&&this.$input.blur()})),this.$input.addEventListener("blur",(()=>{this._callOnFinishChange()})),this.$widget.appendChild(this.$input),this.$disable=this.$input,this.updateDisplay()}updateDisplay(){return this.$input.value=this.getValue(),this}}let g=!1;class m{constructor({parent:t,autoPlace:e=void 0===t,container:i,width:n,title:s="Controls",closeFolders:r=!1,injectStyles:l=!0,touchStyles:o=!0}={}){if(this.parent=t,this.root=t?t.root:this,this.children=[],this.controllers=[],this.folders=[],this._closed=!1,this._hidden=!1,this.domElement=document.createElement("div"),this.domElement.classList.add("lil-gui"),this.$title=document.createElement("div"),this.$title.classList.add("title"),this.$title.setAttribute("role","button"),this.$title.setAttribute("aria-expanded",!0),this.$title.setAttribute("tabindex",0),this.$title.addEventListener("click",(()=>this.openAnimated(this._closed))),this.$title.addEventListener("keydown",(t=>{"Enter"!==t.code&&"Space"!==t.code||(t.preventDefault(),this.$title.click())})),this.$title.addEventListener("touchstart",(()=>{}),{passive:!0}),this.$children=document.createElement("div"),this.$children.classList.add("children"),this.domElement.appendChild(this.$title),this.domElement.appendChild(this.$children),this.title(s),o&&this.domElement.classList.add("allow-touch-styles"),this.parent)return this.parent.children.push(this),this.parent.folders.push(this),void this.parent.$children.appendChild(this.domElement);this.domElement.classList.add("root"),!g&&l&&(function(t){const e=document.createElement("style");e.innerHTML='.lil-gui {\n  font-family: var(--font-family);\n  font-size: var(--font-size);\n  line-height: 1;\n  font-weight: normal;\n  font-style: normal;\n  text-align: left;\n  background-color: var(--background-color);\n  color: var(--text-color);\n  user-select: none;\n  -webkit-user-select: none;\n  touch-action: manipulation;\n  --background-color: #1f1f1f;\n  --text-color: #ebebeb;\n  --title-background-color: #111111;\n  --title-text-color: #ebebeb;\n  --widget-color: #424242;\n  --hover-color: #4f4f4f;\n  --focus-color: #595959;\n  --number-color: #2cc9ff;\n  --string-color: #a2db3c;\n  --font-size: 11px;\n  --input-font-size: 11px;\n  --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;\n  --font-family-mono: Menlo, Monaco, Consolas, "Droid Sans Mono", monospace;\n  --padding: 4px;\n  --spacing: 4px;\n  --widget-height: 20px;\n  --title-height: calc(var(--widget-height) + var(--spacing) * 1.25);\n  --name-width: 45%;\n  --slider-knob-width: 2px;\n  --slider-input-width: 27%;\n  --color-input-width: 27%;\n  --slider-input-min-width: 45px;\n  --color-input-min-width: 45px;\n  --folder-indent: 7px;\n  --widget-padding: 0 0 0 3px;\n  --widget-border-radius: 2px;\n  --checkbox-size: calc(0.75 * var(--widget-height));\n  --scrollbar-width: 5px;\n}\n.lil-gui, .lil-gui * {\n  box-sizing: border-box;\n  margin: 0;\n  padding: 0;\n}\n.lil-gui.root {\n  width: var(--width, 245px);\n  display: flex;\n  flex-direction: column;\n}\n.lil-gui.root > .title {\n  background: var(--title-background-color);\n  color: var(--title-text-color);\n}\n.lil-gui.root > .children {\n  overflow-x: hidden;\n  overflow-y: auto;\n}\n.lil-gui.root > .children::-webkit-scrollbar {\n  width: var(--scrollbar-width);\n  height: var(--scrollbar-width);\n  background: var(--background-color);\n}\n.lil-gui.root > .children::-webkit-scrollbar-thumb {\n  border-radius: var(--scrollbar-width);\n  background: var(--focus-color);\n}\n@media (pointer: coarse) {\n  .lil-gui.allow-touch-styles {\n    --widget-height: 28px;\n    --padding: 6px;\n    --spacing: 6px;\n    --font-size: 13px;\n    --input-font-size: 16px;\n    --folder-indent: 10px;\n    --scrollbar-width: 7px;\n    --slider-input-min-width: 50px;\n    --color-input-min-width: 65px;\n  }\n}\n.lil-gui.force-touch-styles {\n  --widget-height: 28px;\n  --padding: 6px;\n  --spacing: 6px;\n  --font-size: 13px;\n  --input-font-size: 16px;\n  --folder-indent: 10px;\n  --scrollbar-width: 7px;\n  --slider-input-min-width: 50px;\n  --color-input-min-width: 65px;\n}\n.lil-gui.autoPlace {\n  max-height: 100%;\n  position: fixed;\n  top: 0;\n  right: 15px;\n  z-index: 1001;\n}\n\n.lil-gui .controller {\n  display: flex;\n  align-items: center;\n  padding: 0 var(--padding);\n  margin: var(--spacing) 0;\n}\n.lil-gui .controller.disabled {\n  opacity: 0.5;\n}\n.lil-gui .controller.disabled, .lil-gui .controller.disabled * {\n  pointer-events: none !important;\n}\n.lil-gui .controller > .name {\n  min-width: var(--name-width);\n  flex-shrink: 0;\n  white-space: pre;\n  padding-right: var(--spacing);\n  line-height: var(--widget-height);\n}\n.lil-gui .controller .widget {\n  position: relative;\n  display: flex;\n  align-items: center;\n  width: 100%;\n  min-height: var(--widget-height);\n}\n.lil-gui .controller.string input {\n  color: var(--string-color);\n}\n.lil-gui .controller.boolean .widget {\n  cursor: pointer;\n}\n.lil-gui .controller.color .display {\n  width: 100%;\n  height: var(--widget-height);\n  border-radius: var(--widget-border-radius);\n  position: relative;\n}\n@media (hover: hover) {\n  .lil-gui .controller.color .display:hover:before {\n    content: " ";\n    display: block;\n    position: absolute;\n    border-radius: var(--widget-border-radius);\n    border: 1px solid #fff9;\n    top: 0;\n    right: 0;\n    bottom: 0;\n    left: 0;\n  }\n}\n.lil-gui .controller.color input[type=color] {\n  opacity: 0;\n  width: 100%;\n  height: 100%;\n  cursor: pointer;\n}\n.lil-gui .controller.color input[type=text] {\n  margin-left: var(--spacing);\n  font-family: var(--font-family-mono);\n  min-width: var(--color-input-min-width);\n  width: var(--color-input-width);\n  flex-shrink: 0;\n}\n.lil-gui .controller.option select {\n  opacity: 0;\n  position: absolute;\n  width: 100%;\n  max-width: 100%;\n}\n.lil-gui .controller.option .display {\n  position: relative;\n  pointer-events: none;\n  border-radius: var(--widget-border-radius);\n  height: var(--widget-height);\n  line-height: var(--widget-height);\n  max-width: 100%;\n  overflow: hidden;\n  word-break: break-all;\n  padding-left: 0.55em;\n  padding-right: 1.75em;\n  background: var(--widget-color);\n}\n@media (hover: hover) {\n  .lil-gui .controller.option .display.focus {\n    background: var(--focus-color);\n  }\n}\n.lil-gui .controller.option .display.active {\n  background: var(--focus-color);\n}\n.lil-gui .controller.option .display:after {\n  font-family: "lil-gui";\n  content: "↕";\n  position: absolute;\n  top: 0;\n  right: 0;\n  bottom: 0;\n  padding-right: 0.375em;\n}\n.lil-gui .controller.option .widget,\n.lil-gui .controller.option select {\n  cursor: pointer;\n}\n@media (hover: hover) {\n  .lil-gui .controller.option .widget:hover .display {\n    background: var(--hover-color);\n  }\n}\n.lil-gui .controller.number input {\n  color: var(--number-color);\n}\n.lil-gui .controller.number.hasSlider input {\n  margin-left: var(--spacing);\n  width: var(--slider-input-width);\n  min-width: var(--slider-input-min-width);\n  flex-shrink: 0;\n}\n.lil-gui .controller.number .slider {\n  width: 100%;\n  height: var(--widget-height);\n  background-color: var(--widget-color);\n  border-radius: var(--widget-border-radius);\n  padding-right: var(--slider-knob-width);\n  overflow: hidden;\n  cursor: ew-resize;\n  touch-action: pan-y;\n}\n@media (hover: hover) {\n  .lil-gui .controller.number .slider:hover {\n    background-color: var(--hover-color);\n  }\n}\n.lil-gui .controller.number .slider.active {\n  background-color: var(--focus-color);\n}\n.lil-gui .controller.number .slider.active .fill {\n  opacity: 0.95;\n}\n.lil-gui .controller.number .fill {\n  height: 100%;\n  border-right: var(--slider-knob-width) solid var(--number-color);\n  box-sizing: content-box;\n}\n\n.lil-gui-dragging .lil-gui {\n  --hover-color: var(--widget-color);\n}\n.lil-gui-dragging * {\n  cursor: ew-resize !important;\n}\n\n.lil-gui-dragging.lil-gui-vertical * {\n  cursor: ns-resize !important;\n}\n\n.lil-gui .title {\n  height: var(--title-height);\n  line-height: calc(var(--title-height) - 4px);\n  font-weight: 600;\n  padding: 0 var(--padding);\n  -webkit-tap-highlight-color: transparent;\n  cursor: pointer;\n  outline: none;\n  text-decoration-skip: objects;\n}\n.lil-gui .title:before {\n  font-family: "lil-gui";\n  content: "▾";\n  padding-right: 2px;\n  display: inline-block;\n}\n.lil-gui .title:active {\n  background: var(--title-background-color);\n  opacity: 0.75;\n}\n@media (hover: hover) {\n  body:not(.lil-gui-dragging) .lil-gui .title:hover {\n    background: var(--title-background-color);\n    opacity: 0.85;\n  }\n  .lil-gui .title:focus {\n    text-decoration: underline var(--focus-color);\n  }\n}\n.lil-gui.root > .title:focus {\n  text-decoration: none !important;\n}\n.lil-gui.closed > .title:before {\n  content: "▸";\n}\n.lil-gui.closed > .children {\n  transform: translateY(-7px);\n  opacity: 0;\n}\n.lil-gui.closed:not(.transition) > .children {\n  display: none;\n}\n.lil-gui.transition > .children {\n  transition-duration: 300ms;\n  transition-property: height, opacity, transform;\n  transition-timing-function: cubic-bezier(0.2, 0.6, 0.35, 1);\n  overflow: hidden;\n  pointer-events: none;\n}\n.lil-gui .children:empty:before {\n  content: "Empty";\n  padding: 0 var(--padding);\n  margin: var(--spacing) 0;\n  display: block;\n  height: var(--widget-height);\n  font-style: italic;\n  line-height: var(--widget-height);\n  opacity: 0.5;\n}\n.lil-gui.root > .children > .lil-gui > .title {\n  border: 0 solid var(--widget-color);\n  border-width: 1px 0;\n  transition: border-color 300ms;\n}\n.lil-gui.root > .children > .lil-gui.closed > .title {\n  border-bottom-color: transparent;\n}\n.lil-gui + .controller {\n  border-top: 1px solid var(--widget-color);\n  margin-top: 0;\n  padding-top: var(--spacing);\n}\n.lil-gui .lil-gui .lil-gui > .title {\n  border: none;\n}\n.lil-gui .lil-gui .lil-gui > .children {\n  border: none;\n  margin-left: var(--folder-indent);\n  border-left: 2px solid var(--widget-color);\n}\n.lil-gui .lil-gui .controller {\n  border: none;\n}\n\n.lil-gui input {\n  -webkit-tap-highlight-color: transparent;\n  border: 0;\n  outline: none;\n  font-family: var(--font-family);\n  font-size: var(--input-font-size);\n  border-radius: var(--widget-border-radius);\n  height: var(--widget-height);\n  background: var(--widget-color);\n  color: var(--text-color);\n  width: 100%;\n}\n@media (hover: hover) {\n  .lil-gui input:hover {\n    background: var(--hover-color);\n  }\n  .lil-gui input:active {\n    background: var(--focus-color);\n  }\n}\n.lil-gui input:disabled {\n  opacity: 1;\n}\n.lil-gui input[type=text],\n.lil-gui input[type=number] {\n  padding: var(--widget-padding);\n}\n.lil-gui input[type=text]:focus,\n.lil-gui input[type=number]:focus {\n  background: var(--focus-color);\n}\n.lil-gui input::-webkit-outer-spin-button,\n.lil-gui input::-webkit-inner-spin-button {\n  -webkit-appearance: none;\n  margin: 0;\n}\n.lil-gui input[type=number] {\n  -moz-appearance: textfield;\n}\n.lil-gui input[type=checkbox] {\n  appearance: none;\n  -webkit-appearance: none;\n  height: var(--checkbox-size);\n  width: var(--checkbox-size);\n  border-radius: var(--widget-border-radius);\n  text-align: center;\n  cursor: pointer;\n}\n.lil-gui input[type=checkbox]:checked:before {\n  font-family: "lil-gui";\n  content: "✓";\n  font-size: var(--checkbox-size);\n  line-height: var(--checkbox-size);\n}\n@media (hover: hover) {\n  .lil-gui input[type=checkbox]:focus {\n    box-shadow: inset 0 0 0 1px var(--focus-color);\n  }\n}\n.lil-gui button {\n  -webkit-tap-highlight-color: transparent;\n  outline: none;\n  cursor: pointer;\n  font-family: var(--font-family);\n  font-size: var(--font-size);\n  color: var(--text-color);\n  width: 100%;\n  height: var(--widget-height);\n  text-transform: none;\n  background: var(--widget-color);\n  border-radius: var(--widget-border-radius);\n  border: 1px solid var(--widget-color);\n  text-align: center;\n  line-height: calc(var(--widget-height) - 4px);\n}\n@media (hover: hover) {\n  .lil-gui button:hover {\n    background: var(--hover-color);\n    border-color: var(--hover-color);\n  }\n  .lil-gui button:focus {\n    border-color: var(--focus-color);\n  }\n}\n.lil-gui button:active {\n  background: var(--focus-color);\n}\n\n@font-face {\n  font-family: "lil-gui";\n  src: url("data:application/font-woff;charset=utf-8;base64,d09GRgABAAAAAAUsAAsAAAAACJwAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAABHU1VCAAABCAAAAH4AAADAImwmYE9TLzIAAAGIAAAAPwAAAGBKqH5SY21hcAAAAcgAAAD0AAACrukyyJBnbHlmAAACvAAAAF8AAACEIZpWH2hlYWQAAAMcAAAAJwAAADZfcj2zaGhlYQAAA0QAAAAYAAAAJAC5AHhobXR4AAADXAAAABAAAABMAZAAAGxvY2EAAANsAAAAFAAAACgCEgIybWF4cAAAA4AAAAAeAAAAIAEfABJuYW1lAAADoAAAASIAAAIK9SUU/XBvc3QAAATEAAAAZgAAAJCTcMc2eJxVjbEOgjAURU+hFRBK1dGRL+ALnAiToyMLEzFpnPz/eAshwSa97517c/MwwJmeB9kwPl+0cf5+uGPZXsqPu4nvZabcSZldZ6kfyWnomFY/eScKqZNWupKJO6kXN3K9uCVoL7iInPr1X5baXs3tjuMqCtzEuagm/AAlzQgPAAB4nGNgYRBlnMDAysDAYM/gBiT5oLQBAwuDJAMDEwMrMwNWEJDmmsJwgCFeXZghBcjlZMgFCzOiKOIFAB71Bb8AeJy1kjFuwkAQRZ+DwRAwBtNQRUGKQ8OdKCAWUhAgKLhIuAsVSpWz5Bbkj3dEgYiUIszqWdpZe+Z7/wB1oCYmIoboiwiLT2WjKl/jscrHfGg/pKdMkyklC5Zs2LEfHYpjcRoPzme9MWWmk3dWbK9ObkWkikOetJ554fWyoEsmdSlt+uR0pCJR34b6t/TVg1SY3sYvdf8vuiKrpyaDXDISiegp17p7579Gp3p++y7HPAiY9pmTibljrr85qSidtlg4+l25GLCaS8e6rRxNBmsnERunKbaOObRz7N72ju5vdAjYpBXHgJylOAVsMseDAPEP8LYoUHicY2BiAAEfhiAGJgZWBgZ7RnFRdnVJELCQlBSRlATJMoLV2DK4glSYs6ubq5vbKrJLSbGrgEmovDuDJVhe3VzcXFwNLCOILB/C4IuQ1xTn5FPilBTj5FPmBAB4WwoqAHicY2BkYGAA4sk1sR/j+W2+MnAzpDBgAyEMQUCSg4EJxAEAwUgFHgB4nGNgZGBgSGFggJMhDIwMqEAYAByHATJ4nGNgAIIUNEwmAABl3AGReJxjYAACIQYlBiMGJ3wQAEcQBEV4nGNgZGBgEGZgY2BiAAEQyQWEDAz/wXwGAAsPATIAAHicXdBNSsNAHAXwl35iA0UQXYnMShfS9GPZA7T7LgIu03SSpkwzYTIt1BN4Ak/gKTyAeCxfw39jZkjymzcvAwmAW/wgwHUEGDb36+jQQ3GXGot79L24jxCP4gHzF/EIr4jEIe7wxhOC3g2TMYy4Q7+Lu/SHuEd/ivt4wJd4wPxbPEKMX3GI5+DJFGaSn4qNzk8mcbKSR6xdXdhSzaOZJGtdapd4vVPbi6rP+cL7TGXOHtXKll4bY1Xl7EGnPtp7Xy2n00zyKLVHfkHBa4IcJ2oD3cgggWvt/V/FbDrUlEUJhTn/0azVWbNTNr0Ens8de1tceK9xZmfB1CPjOmPH4kitmvOubcNpmVTN3oFJyjzCvnmrwhJTzqzVj9jiSX911FjeAAB4nG3HMRKCMBBA0f0giiKi4DU8k0V2GWbIZDOh4PoWWvq6J5V8If9NVNQcaDhyouXMhY4rPTcG7jwYmXhKq8Wz+p762aNaeYXom2n3m2dLTVgsrCgFJ7OTmIkYbwIbC6vIB7WmFfAAAA==") format("woff");\n}';const i=document.querySelector("head link[rel=stylesheet], head style");i?document.head.insertBefore(e,i):document.head.appendChild(e)}(),g=!0),i?i.appendChild(this.domElement):e&&(this.domElement.classList.add("autoPlace"),document.body.appendChild(this.domElement)),n&&this.domElement.style.setProperty("--width",n+"px"),this._closeFolders=r,this.domElement.addEventListener("keydown",(t=>t.stopPropagation())),this.domElement.addEventListener("keyup",(t=>t.stopPropagation()))}add(t,e,n,s,r){if(Object(n)===n)return new u(this,t,e,n);const l=t[e];switch(typeof l){case"number":return new c(this,t,e,n,s,r);case"boolean":return new i(this,t,e);case"string":return new p(this,t,e);case"function":return new d(this,t,e)}console.error("gui.add failed\n\tproperty:",e,"\n\tobject:",t,"\n\tvalue:",l)}addColor(t,e,i=1){return new h(this,t,e,i)}addFolder(t){const e=new m({parent:this,title:t});return this.root._closeFolders&&e.close(),e}load(t,e=!0){return t.controllers&&this.controllers.forEach((e=>{e instanceof d||e._name in t.controllers&&e.load(t.controllers[e._name])})),e&&t.folders&&this.folders.forEach((e=>{e._title in t.folders&&e.load(t.folders[e._title])})),this}save(t=!0){const e={controllers:{},folders:{}};return this.controllers.forEach((t=>{if(!(t instanceof d)){if(t._name in e.controllers)throw new Error(`Cannot save GUI with duplicate property "${t._name}"`);e.controllers[t._name]=t.save()}})),t&&this.folders.forEach((t=>{if(t._title in e.folders)throw new Error(`Cannot save GUI with duplicate folder "${t._title}"`);e.folders[t._title]=t.save()})),e}open(t=!0){return this._setClosed(!t),this.$title.setAttribute("aria-expanded",!this._closed),this.domElement.classList.toggle("closed",this._closed),this}close(){return this.open(!1)}_setClosed(t){this._closed!==t&&(this._closed=t,this._callOnOpenClose(this))}show(t=!0){return this._hidden=!t,this.domElement.style.display=this._hidden?"none":"",this}hide(){return this.show(!1)}openAnimated(t=!0){return this._setClosed(!t),this.$title.setAttribute("aria-expanded",!this._closed),requestAnimationFrame((()=>{const e=this.$children.clientHeight;this.$children.style.height=e+"px",this.domElement.classList.add("transition");const i=t=>{t.target===this.$children&&(this.$children.style.height="",this.domElement.classList.remove("transition"),this.$children.removeEventListener("transitionend",i))};this.$children.addEventListener("transitionend",i);const n=t?this.$children.scrollHeight:0;this.domElement.classList.toggle("closed",!t),requestAnimationFrame((()=>{this.$children.style.height=n+"px"}))})),this}title(t){return this._title=t,this.$title.innerHTML=t,this}reset(t=!0){return(t?this.controllersRecursive():this.controllers).forEach((t=>t.reset())),this}onChange(t){return this._onChange=t,this}_callOnChange(t){this.parent&&this.parent._callOnChange(t),void 0!==this._onChange&&this._onChange.call(this,{object:t.object,property:t.property,value:t.getValue(),controller:t})}onFinishChange(t){return this._onFinishChange=t,this}_callOnFinishChange(t){this.parent&&this.parent._callOnFinishChange(t),void 0!==this._onFinishChange&&this._onFinishChange.call(this,{object:t.object,property:t.property,value:t.getValue(),controller:t})}onOpenClose(t){return this._onOpenClose=t,this}_callOnOpenClose(t){this.parent&&this.parent._callOnOpenClose(t),void 0!==this._onOpenClose&&this._onOpenClose.call(this,t)}destroy(){this.parent&&(this.parent.children.splice(this.parent.children.indexOf(this),1),this.parent.folders.splice(this.parent.folders.indexOf(this),1)),this.domElement.parentElement&&this.domElement.parentElement.removeChild(this.domElement),Array.from(this.children).forEach((t=>t.destroy()))}controllersRecursive(){let t=Array.from(this.controllers);return this.folders.forEach((e=>{t=t.concat(e.controllersRecursive())})),t}foldersRecursive(){let t=Array.from(this.folders);return this.folders.forEach((e=>{t=t.concat(e.foldersRecursive())})),t}}const b=m;o.e(235).then(o.bind(o,235)).then((e=>{const i=t=>document.getElementById(t),n=new t;n.dom.style.position="absolute";const s=n.addPanel(new t.Panel("MS (Sim)","#ff8","#221"));let r=1;n.showPanel(n.dom.children.length-1),i("container").appendChild(n.dom);const l=new b({autoPlace:!1});l.domElement.style.opacity="0.9";let o={particles:0,viscosity:0,substeps:10,singleColor:!1,block:()=>{g.add_block(),h()},reset10x1000:()=>a(10,1e3),reset10x200:()=>a(10,200),reset40x100:()=>a(40,100),reset100x100:()=>a(100,100)};const a=(t,e)=>{g.reset(t,e),h()},h=()=>{o.particles=g.num_particles,d.updateDisplay()},d=l.add(o,"particles").disable();l.add(o,"viscosity",0,.75,.005).onChange((t=>g.viscosity=t)),l.add(o,"substeps",5,10,1).onChange((t=>g.solver_substeps=t)),l.add(o,"singleColor").name("draw single color").onFinishChange((t=>g.draw_single_color=t)),l.add(o,"block").name("add block"),l.add(o,"reset10x1000").name("reset 10x1000 block"),l.add(o,"reset10x200").name("reset 10x200 block"),l.add(o,"reset40x100").name("reset 40x100 block"),l.add(o,"reset100x100").name("reset 100x100 block"),i("gui").appendChild(l.domElement);const c=window.matchMedia("(prefers-color-scheme: dark)").matches,u=i("canvas");u.width=800,u.height=600;const p=u.getContext("webgl2",{antialias:!0,desynchronized:!0,powerPreference:"high-performance"}),g=new e.Simulation(p,u.width,u.height,c);h();const m=()=>{n.begin();let t=performance.now();g.step(),t=performance.now()-t,g.draw(),s.update(t,r=Math.max(r,t)),n.end(),requestAnimationFrame(m)};requestAnimationFrame(m)})).catch(console.error)})()})();