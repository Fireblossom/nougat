(window.webpackJsonp=window.webpackJsonp||[]).push([[10,11],{354:function(t,e,n){var r=n(358).has;t.exports=function(t){return r(t),t}},356:function(t,e,n){var r=n(4),o=n(415),c=n(358),l=c.Map,d=c.proto,f=r(d.forEach),h=r(d.entries),v=h(new l).next;t.exports=function(map,t,e){return e?o(h(map),(function(e){return t(e[1],e[0])}),v):f(map,t)}},358:function(t,e,n){var r=n(4),o=Map.prototype;t.exports={Map:Map,set:r(o.set),get:r(o.get),has:r(o.has),remove:r(o.delete),proto:o}},359:function(t,e,n){"use strict";n.d(e,"a",(function(){return d})),n.d(e,"b",(function(){return f}));var r=n(375),o=n(10),c=Object(o.h)("v-card__actions"),l=Object(o.h)("v-card__subtitle"),d=Object(o.h)("v-card__text"),f=Object(o.h)("v-card__title");r.a},363:function(t,e,n){"use strict";var r=n(376);e.a=r.a},377:function(t,e,n){var content=n(411);content.__esModule&&(content=content.default),"string"==typeof content&&(content=[[t.i,content,""]]),content.locals&&(t.exports=content.locals);(0,n(62).default)("322289bb",content,!0,{sourceMap:!1})},378:function(t,e,n){"use strict";n.r(e);var r=n(1292),o=n(376),c=n(686),l=n(1290),d=n(458),f={name:"MarkdownDisplay",components:{MathpixMarkdown:d.MathpixMarkdown,MathpixLoader:d.MathpixLoader},props:{source:String,loading:Boolean},data:function(){return{renderMarkdown:!0}}},h=(n(410),n(74)),component=Object(h.a)(f,(function(){var t=this,e=t._self._c;return t.loading?e(c.a,{staticClass:"parent",attrs:{outlined:""}},[e(l.a,{staticClass:"mx-4",attrs:{type:"article, sentences, text@2, sentences, paragraph, sentences, text, paragraph, text, table-thead, table-row-divider@3, table-row, table-heading, text"}})],1):t.source?e(c.a,{staticClass:"parent overflow-y-auto",attrs:{outlined:"","max-height":"800"}},[e(r.a,{staticClass:"upright pa-1",attrs:{tile:"",icon:""},on:{click:function(e){t.renderMarkdown=!t.renderMarkdown}}},[t.renderMarkdown?e(o.a,{attrs:{large:""}},[t._v("mdi-raw")]):e(o.a,{attrs:{large:""}},[t._v("mdi-raw-off")])],1),t._v(" "),t.renderMarkdown?e("MathpixLoader",{staticClass:"serif"},[e("MathpixMarkdown",{attrs:{text:t.source}})],1):e("pre",{attrs:{id:"markdown"}},[t._v(t._s(t.source)+"\n    ")])],1):t._e()}),[],!1,null,null,null);e.default=component.exports},383:function(t,e,n){"use strict";n.d(e,"a",(function(){return l}));var r=n(132);var o=n(169),c=n(110);function l(t){return function(t){if(Array.isArray(t))return Object(r.a)(t)}(t)||Object(o.a)(t)||Object(c.a)(t)||function(){throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}()}},388:function(t,e,n){n(412)},389:function(t,e,n){"use strict";var r=n(2),o=n(354),c=n(358).remove;r({target:"Map",proto:!0,real:!0,forced:!0},{deleteAll:function(){for(var t,e=o(this),n=!0,r=0,l=arguments.length;r<l;r++)t=c(e,arguments[r]),n=n&&t;return!!n}})},390:function(t,e,n){"use strict";var r=n(2),o=n(73),c=n(354),l=n(356);r({target:"Map",proto:!0,real:!0,forced:!0},{every:function(t){var map=c(this),e=o(t,arguments.length>1?arguments[1]:void 0);return!1!==l(map,(function(t,n){if(!e(t,n,map))return!1}),!0)}})},391:function(t,e,n){"use strict";var r=n(2),o=n(73),c=n(354),l=n(358),d=n(356),f=l.Map,h=l.set;r({target:"Map",proto:!0,real:!0,forced:!0},{filter:function(t){var map=c(this),e=o(t,arguments.length>1?arguments[1]:void 0),n=new f;return d(map,(function(t,r){e(t,r,map)&&h(n,r,t)})),n}})},392:function(t,e,n){"use strict";var r=n(2),o=n(73),c=n(354),l=n(356);r({target:"Map",proto:!0,real:!0,forced:!0},{find:function(t){var map=c(this),e=o(t,arguments.length>1?arguments[1]:void 0),n=l(map,(function(t,n){if(e(t,n,map))return{value:t}}),!0);return n&&n.value}})},393:function(t,e,n){"use strict";var r=n(2),o=n(73),c=n(354),l=n(356);r({target:"Map",proto:!0,real:!0,forced:!0},{findKey:function(t){var map=c(this),e=o(t,arguments.length>1?arguments[1]:void 0),n=l(map,(function(t,n){if(e(t,n,map))return{key:n}}),!0);return n&&n.key}})},394:function(t,e,n){"use strict";var r=n(2),o=n(416),c=n(354),l=n(356);r({target:"Map",proto:!0,real:!0,forced:!0},{includes:function(t){return!0===l(c(this),(function(e){if(o(e,t))return!0}),!0)}})},395:function(t,e,n){"use strict";var r=n(2),o=n(354),c=n(356);r({target:"Map",proto:!0,real:!0,forced:!0},{keyOf:function(t){var e=c(o(this),(function(e,n){if(e===t)return{key:n}}),!0);return e&&e.key}})},396:function(t,e,n){"use strict";var r=n(2),o=n(73),c=n(354),l=n(358),d=n(356),f=l.Map,h=l.set;r({target:"Map",proto:!0,real:!0,forced:!0},{mapKeys:function(t){var map=c(this),e=o(t,arguments.length>1?arguments[1]:void 0),n=new f;return d(map,(function(t,r){h(n,e(t,r,map),t)})),n}})},397:function(t,e,n){"use strict";var r=n(2),o=n(73),c=n(354),l=n(358),d=n(356),f=l.Map,h=l.set;r({target:"Map",proto:!0,real:!0,forced:!0},{mapValues:function(t){var map=c(this),e=o(t,arguments.length>1?arguments[1]:void 0),n=new f;return d(map,(function(t,r){h(n,r,e(t,r,map))})),n}})},398:function(t,e,n){"use strict";var r=n(2),o=n(354),c=n(161),l=n(358).set;r({target:"Map",proto:!0,real:!0,arity:1,forced:!0},{merge:function(t){for(var map=o(this),e=arguments.length,i=0;i<e;)c(arguments[i++],(function(t,e){l(map,t,e)}),{AS_ENTRIES:!0});return map}})},399:function(t,e,n){"use strict";var r=n(2),o=n(43),c=n(354),l=n(356),d=TypeError;r({target:"Map",proto:!0,real:!0,forced:!0},{reduce:function(t){var map=c(this),e=arguments.length<2,n=e?void 0:arguments[1];if(o(t),l(map,(function(r,o){e?(e=!1,n=r):n=t(n,r,o,map)})),e)throw d("Reduce of empty map with no initial value");return n}})},400:function(t,e,n){"use strict";var r=n(2),o=n(73),c=n(354),l=n(356);r({target:"Map",proto:!0,real:!0,forced:!0},{some:function(t){var map=c(this),e=o(t,arguments.length>1?arguments[1]:void 0);return!0===l(map,(function(t,n){if(e(t,n,map))return!0}),!0)}})},401:function(t,e,n){"use strict";var r=n(2),o=n(43),c=n(354),l=n(358),d=TypeError,f=l.get,h=l.has,v=l.set;r({target:"Map",proto:!0,real:!0,forced:!0},{update:function(t,e){var map=c(this),n=arguments.length;o(e);var r=h(map,t);if(!r&&n<3)throw d("Updating absent value");var l=r?f(map,t):o(n>2?arguments[2]:void 0)(t,map);return v(map,t,e(l,t,map)),map}})},404:function(t,e,n){"use strict";var r={inserted:function(t,e,n){var r=e.value,o=e.options||{passive:!0};window.addEventListener("resize",r,o),t._onResize=Object(t._onResize),t._onResize[n.context._uid]={callback:r,options:o},e.modifiers&&e.modifiers.quiet||r()},unbind:function(t,e,n){var r;if(null===(r=t._onResize)||void 0===r?void 0:r[n.context._uid]){var o=t._onResize[n.context._uid],c=o.callback,l=o.options;window.removeEventListener("resize",c,l),delete t._onResize[n.context._uid]}}};e.a=r},410:function(t,e,n){"use strict";n(377)},411:function(t,e,n){var r=n(61)((function(i){return i[1]}));r.push([t.i,'@font-face{font-family:"Computer Modern";src:"assets/cmunss.otf"}@font-face{font-family:"Computer Modern";font-weight:700;src:"assets/cmunsx.otf"}@font-face{font-family:"Computer Modern";font-style:italic,oblique;src:"assets/cmunsi.otf"}@font-face{font-family:"Computer Modern";font-style:italic,oblique;font-weight:700;src:"assets/cmunbxo.otf"}.parent{height:100%;width:100%}h1{font-size:22px}h2{font-size:20px}h3{font-size:18px}h4,h5,h6{font-size:16px}.upright{float:right;position:sticky;right:5px;top:5px;z-index:5}#markdown{word-wrap:break-word;white-space:pre-wrap;white-space:-moz-pre-wrap;white-space:-pre-wrap;white-space:-o-pre-wrap}.serif *{font-family:"Computer Modern",serif}',""]),r.locals={},t.exports=r},412:function(t,e,n){"use strict";n(413)("Map",(function(t){return function(){return t(this,arguments.length?arguments[0]:void 0)}}),n(414))},413:function(t,e,n){"use strict";var r=n(2),o=n(7),c=n(4),l=n(108),d=n(29),f=n(240),h=n(161),v=n(162),m=n(6),y=n(51),x=n(21),_=n(3),w=n(164),j=n(88),S=n(168);t.exports=function(t,e,n){var O=-1!==t.indexOf("Map"),k=-1!==t.indexOf("Weak"),M=O?"set":"add",z=o[t],C=z&&z.prototype,D=z,E={},L=function(t){var e=c(C[t]);d(C,t,"add"==t?function(t){return e(this,0===t?0:t),this}:"delete"==t?function(t){return!(k&&!x(t))&&e(this,0===t?0:t)}:"get"==t?function(t){return k&&!x(t)?void 0:e(this,0===t?0:t)}:"has"==t?function(t){return!(k&&!x(t))&&e(this,0===t?0:t)}:function(t,n){return e(this,0===t?0:t,n),this})};if(l(t,!m(z)||!(k||C.forEach&&!_((function(){(new z).entries().next()})))))D=n.getConstructor(e,t,O,M),f.enable();else if(l(t,!0)){var N=new D,P=N[M](k?{}:-0,1)!=N,$=_((function(){N.has(1)})),R=w((function(t){new z(t)})),A=!k&&_((function(){for(var t=new z,e=5;e--;)t[M](e,e);return!t.has(-0)}));R||((D=e((function(t,e){v(t,C);var n=S(new z,t,D);return y(e)||h(e,n[M],{that:n,AS_ENTRIES:O}),n}))).prototype=C,C.constructor=D),($||A)&&(L("delete"),L("has"),O&&L("get")),(A||P)&&L(M),k&&C.clear&&delete C.clear}return E[t]=D,r({global:!0,constructor:!0,forced:D!=z},E),j(D,t),k||n.setStrong(D,t,O),D}},414:function(t,e,n){"use strict";var r=n(67),o=n(87),c=n(242),l=n(73),d=n(162),f=n(51),h=n(161),v=n(165),m=n(166),y=n(167),x=n(12),_=n(240).fastKey,w=n(53),j=w.set,S=w.getterFor;t.exports={getConstructor:function(t,e,n,v){var m=t((function(t,o){d(t,y),j(t,{type:e,index:r(null),first:void 0,last:void 0,size:0}),x||(t.size=0),f(o)||h(o,t[v],{that:t,AS_ENTRIES:n})})),y=m.prototype,w=S(e),O=function(t,e,n){var r,o,c=w(t),l=k(t,e);return l?l.value=n:(c.last=l={index:o=_(e,!0),key:e,value:n,previous:r=c.last,next:void 0,removed:!1},c.first||(c.first=l),r&&(r.next=l),x?c.size++:t.size++,"F"!==o&&(c.index[o]=l)),t},k=function(t,e){var n,r=w(t),o=_(e);if("F"!==o)return r.index[o];for(n=r.first;n;n=n.next)if(n.key==e)return n};return c(y,{clear:function(){for(var t=w(this),data=t.index,e=t.first;e;)e.removed=!0,e.previous&&(e.previous=e.previous.next=void 0),delete data[e.index],e=e.next;t.first=t.last=void 0,x?t.size=0:this.size=0},delete:function(t){var e=this,n=w(e),r=k(e,t);if(r){var o=r.next,c=r.previous;delete n.index[r.index],r.removed=!0,c&&(c.next=o),o&&(o.previous=c),n.first==r&&(n.first=o),n.last==r&&(n.last=c),x?n.size--:e.size--}return!!r},forEach:function(t){for(var e,n=w(this),r=l(t,arguments.length>1?arguments[1]:void 0);e=e?e.next:n.first;)for(r(e.value,e.key,this);e&&e.removed;)e=e.previous},has:function(t){return!!k(this,t)}}),c(y,n?{get:function(t){var e=k(this,t);return e&&e.value},set:function(t,e){return O(this,0===t?0:t,e)}}:{add:function(t){return O(this,t=0===t?0:t,t)}}),x&&o(y,"size",{configurable:!0,get:function(){return w(this).size}}),m},setStrong:function(t,e,n){var r=e+" Iterator",o=S(e),c=S(r);v(t,e,(function(t,e){j(this,{type:r,target:t,state:o(t),kind:e,last:void 0})}),(function(){for(var t=c(this),e=t.kind,n=t.last;n&&n.removed;)n=n.previous;return t.target&&(t.last=n=n?n.next:t.state.first)?m("keys"==e?n.key:"values"==e?n.value:[n.key,n.value],!1):(t.target=void 0,m(void 0,!0))}),n?"entries":"values",!n,!0),y(e)}}},415:function(t,e,n){var r=n(9);t.exports=function(t,e,n){for(var o,c,l=n||t.next;!(o=r(l,t)).done;)if(void 0!==(c=e(o.value)))return c}},416:function(t,e){t.exports=function(t,e){return t===e||t!=t&&e!=e}},455:function(t,e,n){var content=n(556);content.__esModule&&(content=content.default),"string"==typeof content&&(content=[[t.i,content,""]]),content.locals&&(t.exports=content.locals);(0,n(62).default)("b0b6fb54",content,!0,{sourceMap:!1})},473:function(t,e){},474:function(t,e){},475:function(t,e){},541:function(t,e,n){"use strict";n(26),n(27),n(38),n(39);var r=n(11),o=(n(5),n(63),n(64),n(32),n(22),n(23),n(47),n(388),n(41),n(389),n(390),n(391),n(392),n(393),n(394),n(395),n(396),n(397),n(398),n(399),n(400),n(401),n(42),n(25),n(241),n(0)),c=n(163),l=n(10);function d(object,t){var e=Object.keys(object);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(object);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(object,t).enumerable}))),e.push.apply(e,n)}return e}function f(t){for(var i=1;i<arguments.length;i++){var source=null!=arguments[i]?arguments[i]:{};i%2?d(Object(source),!0).forEach((function(e){Object(r.a)(t,e,source[e])})):Object.getOwnPropertyDescriptors?Object.defineProperties(t,Object.getOwnPropertyDescriptors(source)):d(Object(source)).forEach((function(e){Object.defineProperty(t,e,Object.getOwnPropertyDescriptor(source,e))}))}return t}var h=["sm","md","lg","xl"],v=["start","end","center"];function m(t,e){return h.reduce((function(n,r){return n[t+Object(l.u)(r)]=e(),n}),{})}var y=function(t){return[].concat(v,["baseline","stretch"]).includes(t)},x=m("align",(function(){return{type:String,default:null,validator:y}})),_=function(t){return[].concat(v,["space-between","space-around"]).includes(t)},w=m("justify",(function(){return{type:String,default:null,validator:_}})),j=function(t){return[].concat(v,["space-between","space-around","stretch"]).includes(t)},S=m("alignContent",(function(){return{type:String,default:null,validator:j}})),O={align:Object.keys(x),justify:Object.keys(w),alignContent:Object.keys(S)},k={align:"align",justify:"justify",alignContent:"align-content"};function M(t,e,n){var r=k[t];if(null!=n){if(e){var o=e.replace(t,"");r+="-".concat(o)}return(r+="-".concat(n)).toLowerCase()}}var z=new Map;e.a=o.default.extend({name:"v-row",functional:!0,props:f(f(f({tag:{type:String,default:"div"},dense:Boolean,noGutters:Boolean,align:{type:String,default:null,validator:y}},x),{},{justify:{type:String,default:null,validator:_}},w),{},{alignContent:{type:String,default:null,validator:j}},S),render:function(t,e){var n=e.props,data=e.data,o=e.children,l="";for(var d in n)l+=String(n[d]);var f=z.get(l);if(!f){var h,v;for(v in f=[],O)O[v].forEach((function(t){var e=n[t],r=M(v,t,e);r&&f.push(r)}));f.push((h={"no-gutters":n.noGutters,"row--dense":n.dense},Object(r.a)(h,"align-".concat(n.align),n.align),Object(r.a)(h,"justify-".concat(n.justify),n.justify),Object(r.a)(h,"align-content-".concat(n.alignContent),n.alignContent),h)),z.set(l,f)}return t(n.tag,Object(c.a)(data,{staticClass:"row",class:f}),o)}})},548:function(t,e,n){n(2)({target:"Number",stat:!0,nonConfigurable:!0,nonWritable:!0},{MAX_SAFE_INTEGER:9007199254740991})},549:function(t,e,n){var content=n(550);content.__esModule&&(content=content.default),"string"==typeof content&&(content=[[t.i,content,""]]),content.locals&&(t.exports=content.locals);(0,n(62).default)("5bec9846",content,!0,{sourceMap:!1})},550:function(t,e,n){var r=n(61)((function(i){return i[1]}));r.push([t.i,".theme--light.v-pagination .v-pagination__item{background:#fff;color:rgba(0,0,0,.87)}.theme--light.v-pagination .v-pagination__item--active{color:#fff}.theme--light.v-pagination .v-pagination__navigation{background:#fff}.theme--dark.v-pagination .v-pagination__item{background:#1e1e1e;color:#fff}.theme--dark.v-pagination .v-pagination__item--active{color:#fff}.theme--dark.v-pagination .v-pagination__navigation{background:#1e1e1e}.v-pagination{align-items:center;display:inline-flex;justify-content:center;list-style-type:none;margin:0;max-width:100%;width:100%}.v-pagination.v-pagination{padding-left:0}.v-pagination>li{align-items:center;display:flex}.v-pagination--circle .v-pagination__item,.v-pagination--circle .v-pagination__more,.v-pagination--circle .v-pagination__navigation{border-radius:50%}.v-pagination--disabled{opacity:.6;pointer-events:none}.v-pagination__item{background:transparent;border-radius:4px;box-shadow:0 3px 1px -2px rgba(0,0,0,.2),0 2px 2px 0 rgba(0,0,0,.14),0 1px 5px 0 rgba(0,0,0,.12);font-size:1rem;height:34px;margin:.3rem;min-width:34px;padding:0 5px;-webkit-text-decoration:none;text-decoration:none;transition:.3s cubic-bezier(0,0,.2,1);width:auto}.v-pagination__item--active{box-shadow:0 2px 4px -1px rgba(0,0,0,.2),0 4px 5px 0 rgba(0,0,0,.14),0 1px 10px 0 rgba(0,0,0,.12)}.v-pagination__navigation{align-items:center;border-radius:4px;box-shadow:0 3px 1px -2px rgba(0,0,0,.2),0 2px 2px 0 rgba(0,0,0,.14),0 1px 5px 0 rgba(0,0,0,.12);display:inline-flex;height:32px;justify-content:center;margin:.3rem 10px;-webkit-text-decoration:none;text-decoration:none;width:32px}.v-pagination__navigation .v-icon{transition:.2s cubic-bezier(.4,0,.6,1);vertical-align:middle}.v-pagination__navigation--disabled{opacity:.6;pointer-events:none}.v-pagination__more{align-items:flex-end;display:inline-flex;height:32px;justify-content:center;margin:.3rem;width:32px}",""]),r.locals={},t.exports=r},551:function(t,e,n){var content=n(552);content.__esModule&&(content=content.default),"string"==typeof content&&(content=[[t.i,content,""]]),content.locals&&(t.exports=content.locals);(0,n(62).default)("a4669b52",content,!0,{sourceMap:!1})},552:function(t,e,n){var r=n(61)((function(i){return i[1]}));r.push([t.i,".theme--light.v-image{color:rgba(0,0,0,.87)}.theme--dark.v-image{color:#fff}.v-image{z-index:0}.v-image__image,.v-image__placeholder{height:100%;left:0;position:absolute;top:0;width:100%;z-index:-1}.v-image__image{background-repeat:no-repeat}.v-image__image--preload{filter:blur(2px)}.v-image__image--contain{background-size:contain}.v-image__image--cover{background-size:cover}",""]),r.locals={},t.exports=r},553:function(t,e,n){var content=n(554);content.__esModule&&(content=content.default),"string"==typeof content&&(content=[[t.i,content,""]]),content.locals&&(t.exports=content.locals);(0,n(62).default)("0c396eac",content,!0,{sourceMap:!1})},554:function(t,e,n){var r=n(61)((function(i){return i[1]}));r.push([t.i,".v-responsive{display:flex;flex:1 0 auto;max-width:100%;overflow:hidden;position:relative}.v-responsive__content{flex:1 0 0px;max-width:100%}.v-application--is-ltr .v-responsive__sizer~.v-responsive__content{margin-left:-100%}.v-application--is-rtl .v-responsive__sizer~.v-responsive__content{margin-right:-100%}.v-responsive__sizer{flex:1 0 0px;transition:padding-bottom .2s cubic-bezier(.25,.8,.5,1)}",""]),r.locals={},t.exports=r},555:function(t,e,n){"use strict";n(455)},556:function(t,e,n){var r=n(61)((function(i){return i[1]}));r.push([t.i,".img-outline{background-color:#e0e0e0;padding:1px}",""]),r.locals={},t.exports=r},557:function(t,e,n){"use strict";n.r(e);var r=n(359),o=(n(26),n(27),n(38),n(39),n(11)),c=(n(5),n(159),n(22),n(23),n(47),n(388),n(41),n(389),n(390),n(391),n(392),n(393),n(394),n(395),n(396),n(397),n(398),n(399),n(400),n(401),n(42),n(63),n(25),n(55),n(241),n(0)),l=n(163),d=n(10);function f(object,t){var e=Object.keys(object);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(object);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(object,t).enumerable}))),e.push.apply(e,n)}return e}function h(t){for(var i=1;i<arguments.length;i++){var source=null!=arguments[i]?arguments[i]:{};i%2?f(Object(source),!0).forEach((function(e){Object(o.a)(t,e,source[e])})):Object.getOwnPropertyDescriptors?Object.defineProperties(t,Object.getOwnPropertyDescriptors(source)):f(Object(source)).forEach((function(e){Object.defineProperty(t,e,Object.getOwnPropertyDescriptor(source,e))}))}return t}var v=["sm","md","lg","xl"],m=v.reduce((function(t,e){return t[e]={type:[Boolean,String,Number],default:!1},t}),{}),y=v.reduce((function(t,e){return t["offset"+Object(d.u)(e)]={type:[String,Number],default:null},t}),{}),x=v.reduce((function(t,e){return t["order"+Object(d.u)(e)]={type:[String,Number],default:null},t}),{}),_={col:Object.keys(m),offset:Object.keys(y),order:Object.keys(x)};function w(t,e,n){var r=t;if(null!=n&&!1!==n){if(e){var o=e.replace(t,"");r+="-".concat(o)}return"col"!==t||""!==n&&!0!==n?(r+="-".concat(n)).toLowerCase():r.toLowerCase()}}var j=new Map,S=c.default.extend({name:"v-col",functional:!0,props:h(h(h(h({cols:{type:[Boolean,String,Number],default:!1}},m),{},{offset:{type:[String,Number],default:null}},y),{},{order:{type:[String,Number],default:null}},x),{},{alignSelf:{type:String,default:null,validator:function(t){return["auto","start","end","center","baseline","stretch"].includes(t)}},tag:{type:String,default:"div"}}),render:function(t,e){var n=e.props,data=e.data,r=e.children,c=(e.parent,"");for(var d in n)c+=String(n[d]);var f=j.get(c);if(!f){var h,v;for(v in f=[],_)_[v].forEach((function(t){var e=n[t],r=w(v,t,e);r&&f.push(r)}));var m=f.some((function(t){return t.startsWith("col-")}));f.push((h={col:!m||!n.cols},Object(o.a)(h,"col-".concat(n.cols),n.cols),Object(o.a)(h,"offset-".concat(n.offset),n.offset),Object(o.a)(h,"order-".concat(n.order),n.order),Object(o.a)(h,"align-self-".concat(n.alignSelf),n.alignSelf),h)),j.set(c,f)}return t(n.tag,Object(l.a)(data,{class:f}),r)}}),O=n(353),k=n(376),M=n(17),z=(n(90),n(247),n(65),n(551),n(422)),C=(n(553),n(456)),D=n(158),E=Object(D.a)(C.a).extend({name:"v-responsive",props:{aspectRatio:[String,Number],contentClass:String},computed:{computedAspectRatio:function(){return Number(this.aspectRatio)},aspectStyle:function(){return this.computedAspectRatio?{paddingBottom:1/this.computedAspectRatio*100+"%"}:void 0},__cachedSizer:function(){return this.aspectStyle?this.$createElement("div",{style:this.aspectStyle,staticClass:"v-responsive__sizer"}):[]}},methods:{genContent:function(){return this.$createElement("div",{staticClass:"v-responsive__content",class:this.contentClass},Object(d.m)(this))}},render:function(t){return t("div",{staticClass:"v-responsive",style:this.measurableStyles,on:this.$listeners},[this.__cachedSizer,this.genContent()])}}),L=n(160),N=n(40),P="undefined"!=typeof window&&"IntersectionObserver"in window,$=Object(D.a)(E,L.a).extend({name:"v-img",directives:{intersect:z.a},props:{alt:String,contain:Boolean,eager:Boolean,gradient:String,lazySrc:String,options:{type:Object,default:function(){return{root:void 0,rootMargin:void 0,threshold:void 0}}},position:{type:String,default:"center center"},sizes:String,src:{type:[String,Object],default:""},srcset:String,transition:{type:[Boolean,String],default:"fade-transition"}},data:function(){return{currentSrc:"",image:null,isLoading:!0,calculatedAspectRatio:void 0,naturalWidth:void 0,hasError:!1}},computed:{computedAspectRatio:function(){return Number(this.normalisedSrc.aspect||this.calculatedAspectRatio)},normalisedSrc:function(){return this.src&&"object"===Object(M.a)(this.src)?{src:this.src.src,srcset:this.srcset||this.src.srcset,lazySrc:this.lazySrc||this.src.lazySrc,aspect:Number(this.aspectRatio||this.src.aspect)}:{src:this.src,srcset:this.srcset,lazySrc:this.lazySrc,aspect:Number(this.aspectRatio||0)}},__cachedImage:function(){if(!(this.normalisedSrc.src||this.normalisedSrc.lazySrc||this.gradient))return[];var t=[],e=this.isLoading?this.normalisedSrc.lazySrc:this.currentSrc;this.gradient&&t.push("linear-gradient(".concat(this.gradient,")")),e&&t.push('url("'.concat(e,'")'));var image=this.$createElement("div",{staticClass:"v-image__image",class:{"v-image__image--preload":this.isLoading,"v-image__image--contain":this.contain,"v-image__image--cover":!this.contain},style:{backgroundImage:t.join(", "),backgroundPosition:this.position},key:+this.isLoading});return this.transition?this.$createElement("transition",{attrs:{name:this.transition,mode:"in-out"}},[image]):image}},watch:{src:function(){this.isLoading?this.loadImage():this.init(void 0,void 0,!0)},"$vuetify.breakpoint.width":"getSrc"},mounted:function(){this.init()},methods:{init:function(t,e,n){if(!P||n||this.eager){if(this.normalisedSrc.lazySrc){var r=new Image;r.src=this.normalisedSrc.lazySrc,this.pollForSize(r,null)}this.normalisedSrc.src&&this.loadImage()}},onLoad:function(){this.getSrc(),this.isLoading=!1,this.$emit("load",this.src),this.image&&(this.normalisedSrc.src.endsWith(".svg")||this.normalisedSrc.src.startsWith("data:image/svg+xml"))&&(this.image.naturalHeight&&this.image.naturalWidth?(this.naturalWidth=this.image.naturalWidth,this.calculatedAspectRatio=this.image.naturalWidth/this.image.naturalHeight):this.calculatedAspectRatio=1)},onError:function(){this.hasError=!0,this.$emit("error",this.src)},getSrc:function(){this.image&&(this.currentSrc=this.image.currentSrc||this.image.src)},loadImage:function(){var t=this,image=new Image;this.image=image,image.onload=function(){image.decode?image.decode().catch((function(e){Object(N.c)("Failed to decode image, trying to render anyway\n\n"+"src: ".concat(t.normalisedSrc.src)+(e.message?"\nOriginal error: ".concat(e.message):""),t)})).then(t.onLoad):t.onLoad()},image.onerror=this.onError,this.hasError=!1,this.sizes&&(image.sizes=this.sizes),this.normalisedSrc.srcset&&(image.srcset=this.normalisedSrc.srcset),image.src=this.normalisedSrc.src,this.$emit("loadstart",this.normalisedSrc.src),this.aspectRatio||this.pollForSize(image),this.getSrc()},pollForSize:function(img){var t=this,e=arguments.length>1&&void 0!==arguments[1]?arguments[1]:100;!function n(){var r=img.naturalHeight,o=img.naturalWidth;r||o?(t.naturalWidth=o,t.calculatedAspectRatio=o/r):img.complete||!t.isLoading||t.hasError||null==e||setTimeout(n,e)}()},genContent:function(){var content=E.options.methods.genContent.call(this);return this.naturalWidth&&this._b(content.data,"div",{style:{width:"".concat(this.naturalWidth,"px")}}),content},__genPlaceholder:function(){var slot=Object(d.m)(this,"placeholder");if(slot){var t=this.isLoading?[this.$createElement("div",{staticClass:"v-image__placeholder"},slot)]:[];return this.transition?this.$createElement("transition",{props:{appear:!0,name:this.transition}},t):t[0]}}},render:function(t){var e=E.options.render.call(this,t),data=Object(l.a)(e.data,{staticClass:"v-image",attrs:{"aria-label":this.alt,role:this.alt?"img":void 0},class:this.themeClasses,directives:P?[{name:"intersect",modifiers:{once:!0},value:{handler:this.init,options:this.options}}]:void 0});return e.children=[this.__cachedSizer,this.__cachedImage,this.__genPlaceholder(),this.genContent()],t(e.tag,data,e.children)}}),R=n(383),A=(n(548),n(32),n(75),n(48),n(549),n(363)),I=n(404),T=n(372);function W(object,t){var e=Object.keys(object);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(object);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(object,t).enumerable}))),e.push.apply(e,n)}return e}var B,F=Object(D.a)(T.a,(B={onVisible:["init"]},c.default.extend({name:"intersectable",data:function(){return{isIntersecting:!1}},mounted:function(){z.a.inserted(this.$el,{name:"intersect",value:this.onObserve},this.$vnode)},destroyed:function(){z.a.unbind(this.$el,{name:"intersect",value:this.onObserve},this.$vnode)},methods:{onObserve:function(t,e,n){if(this.isIntersecting=n,n)for(var i=0,r=B.onVisible.length;i<r;i++){var o=this[B.onVisible[i]];"function"!=typeof o?Object(N.c)(B.onVisible[i]+" method is not available on the instance but referenced in intersectable mixin options"):o()}}}})),L.a).extend({name:"v-pagination",directives:{Resize:I.a},props:{circle:Boolean,disabled:Boolean,length:{type:Number,default:0,validator:function(t){return t%1==0}},nextIcon:{type:String,default:"$next"},prevIcon:{type:String,default:"$prev"},totalVisible:[Number,String],value:{type:Number,default:0},pageAriaLabel:{type:String,default:"$vuetify.pagination.ariaLabel.page"},currentPageAriaLabel:{type:String,default:"$vuetify.pagination.ariaLabel.currentPage"},previousAriaLabel:{type:String,default:"$vuetify.pagination.ariaLabel.previous"},nextAriaLabel:{type:String,default:"$vuetify.pagination.ariaLabel.next"},wrapperAriaLabel:{type:String,default:"$vuetify.pagination.ariaLabel.wrapper"}},data:function(){return{maxButtons:0,selected:null}},computed:{classes:function(){return function(t){for(var i=1;i<arguments.length;i++){var source=null!=arguments[i]?arguments[i]:{};i%2?W(Object(source),!0).forEach((function(e){Object(o.a)(t,e,source[e])})):Object.getOwnPropertyDescriptors?Object.defineProperties(t,Object.getOwnPropertyDescriptors(source)):W(Object(source)).forEach((function(e){Object.defineProperty(t,e,Object.getOwnPropertyDescriptor(source,e))}))}return t}({"v-pagination":!0,"v-pagination--circle":this.circle,"v-pagination--disabled":this.disabled},this.themeClasses)},items:function(){var t=parseInt(this.totalVisible,10);if(0===t||isNaN(this.length)||this.length>Number.MAX_SAFE_INTEGER)return[];var e=Math.min(Math.max(0,t)||this.length,Math.max(0,this.maxButtons)||this.length,this.length);if(this.length<=e)return this.range(1,this.length);var n=e%2==0?1:0,r=Math.floor(e/2),o=this.length-r+1+n;if(this.value>r&&this.value<o){var c=this.length,l=this.value-r+2,d=this.value+r-2-n,f=d+1===c-1?d+1:"...";return[1,l-1==2?2:"..."].concat(Object(R.a)(this.range(l,d)),[f,this.length])}if(this.value===r){var h=this.value+r-1-n;return[].concat(Object(R.a)(this.range(1,h)),["...",this.length])}if(this.value===o){var v=this.value-r+1;return[1,"..."].concat(Object(R.a)(this.range(v,this.length)))}return[].concat(Object(R.a)(this.range(1,r)),["..."],Object(R.a)(this.range(o,this.length)))}},watch:{value:function(){this.init()}},beforeMount:function(){this.init()},methods:{init:function(){var t=this;this.selected=null,this.onResize(),this.$nextTick(this.onResize),setTimeout((function(){return t.selected=t.value}),100)},onResize:function(){var t=this.$el&&this.$el.parentElement?this.$el.parentElement.clientWidth:window.innerWidth;this.maxButtons=Math.floor((t-96)/42)},next:function(t){t.preventDefault(),this.$emit("input",this.value+1),this.$emit("next")},previous:function(t){t.preventDefault(),this.$emit("input",this.value-1),this.$emit("previous")},range:function(t,e){for(var n=[],i=t=t>0?t:1;i<=e;i++)n.push(i);return n},genIcon:function(t,e,n,r,label){return t("li",[t("button",{staticClass:"v-pagination__navigation",class:{"v-pagination__navigation--disabled":n},attrs:{disabled:n,type:"button","aria-label":label},on:n?{}:{click:r}},[t(A.a,[e])])])},genItem:function(t,i){var e=this,n=i===this.value&&(this.color||"primary"),r=i===this.value,o=r?this.currentPageAriaLabel:this.pageAriaLabel;return t("button",this.setBackgroundColor(n,{staticClass:"v-pagination__item",class:{"v-pagination__item--active":i===this.value},attrs:{type:"button","aria-current":r,"aria-label":this.$vuetify.lang.t(o,i)},on:{click:function(){return e.$emit("input",i)}}}),[i.toString()])},genItems:function(t){var e=this;return this.items.map((function(i,n){return t("li",{key:n},[isNaN(Number(i))?t("span",{class:"v-pagination__more"},[i.toString()]):e.genItem(t,i)])}))},genList:function(t,e){return t("ul",{directives:[{modifiers:{quiet:!0},name:"resize",value:this.onResize}],class:this.classes},e)}},render:function(t){var e=[this.genIcon(t,this.$vuetify.rtl?this.nextIcon:this.prevIcon,this.value<=1,this.previous,this.$vuetify.lang.t(this.previousAriaLabel)),this.genItems(t),this.genIcon(t,this.$vuetify.rtl?this.prevIcon:this.nextIcon,this.value>=this.length,this.next,this.$vuetify.lang.t(this.nextAriaLabel))];return t("nav",{attrs:{role:"navigation","aria-label":this.$vuetify.lang.t(this.wrapperAriaLabel)}},[this.genList(t,e)])}}),V=n(541),G=n(686),H=(n(28),n(484),n(18)),K=(n(86),{name:"ScannedPapers",components:{MarkdownDisplay:n(378).default},data:function(){return{loading:!0,page:1,papers:[{img:"/nougat/pages/01.jpg",thumb:"/nougat/pages/thumbs/01.jpg",code:"/nougat/pages/01.mmd",markdown:"and the rule is proved that\r\n\r\n\\[\\frac{du^{*}}{dx}=nw^{*-1}\\frac{du}{dx},\\]\r\n\r\nwhere \\(n\\) is a positive fraction whose numerator and denominator are integers. This rule has already been used in the solution of numerous exercises.\r\n\r\n## 34 The Derivative of a Constant\r\n\r\nLet \\(y=c\\), where \\(c\\) is a constant. Corresponding to any \\(\\Delta x,\\Delta y=0\\), and consequently\r\n\r\n\\[\\frac{\\Delta y}{\\Delta x}=0,\\]\r\n\r\nand\r\n\r\n\\[\\lim\\limits_{\\Delta x\\to 0}\\frac{\\Delta y}{\\Delta x}=0,\\]\r\n\r\nor\r\n\r\n\\[\\frac{dy}{dx}=0.\\]\r\n\r\nThe derivative of a constant is zero.\r\n\r\nInterpret this result geometrically.\r\n\r\n## 35 The Derivative of the Sum of Two Functions\r\n\r\nLet\r\n\r\n\\[y=u+v,\\]\r\n\r\nwhere \\(u\\) and \\(v\\)are functions of \\(x\\).Let \\(\\Delta u\\), \\(\\Delta v\\), and \\(\\Delta y\\) be the increments of \\(u,v\\), and \\(y\\), respectively, corresponding to the increment \\(\\Delta x\\).\r\n\r\n\\[y+\\Delta y =u+\\Delta u+v+\\Delta v\\] \\[\\Delta y =\\Delta u+\\Delta v\\] \\[\\frac{\\Delta y}{\\Delta x} =\\frac{\\Delta u}{\\Delta x}+\\frac{\\Delta v}{\\Delta x}\\] \\[\\frac{dy}{\\bar{dx}} =\\frac{du}{dx}+\\frac{dv}{dx},\\]\r\n\r\nor\r\n\r\n\\[\\frac{d(u+v)}{dx}=\\frac{du}{dx}+\\frac{dv}{dx}.\\]\r\n\r\nThe derivative of the sum of two functions is equal to the sum of their derivatives.",pageNum:43,name:"Calculus (1917) - March, Wolff",link:"https://archive.org/details/calculus00marciala/page/43/mode/1up"},{img:"/nougat/pages/02.jpg",thumb:"/nougat/pages/thumbs/02.jpg",code:"/nougat/pages/02.mmd",markdown:null,pageNum:116,name:"Calculus (1917) - March, Wolff",link:"https://archive.org/details/calculus00marciala/page/116/mode/1up"},{img:"/nougat/pages/03.jpg",thumb:"/nougat/pages/thumbs/03.jpg",code:"/nougat/pages/03.mmd",markdown:null,pageNum:106,name:"Kinetics and Thermodynamics in High-Temperature Gases (1970) - NASA",link:"https://ntrs.nasa.gov/citations/19700022795"},{img:"/nougat/pages/04.jpg",thumb:"/nougat/pages/thumbs/04.jpg",code:"/nougat/pages/04.mmd",markdown:null,pageNum:107,name:"Kinetics and Thermodynamics in High-Temperature Gases (1970) - NASA",link:"https://ntrs.nasa.gov/citations/19700022795"},{img:"/nougat/pages/05.jpg",thumb:"/nougat/pages/thumbs/05.jpg",code:"/nougat/pages/05.mmd",markdown:null,pageNum:3,name:"Scanned Master Thesis (2023) - Wallner"},{img:"/nougat/pages/06.jpg",thumb:"/nougat/pages/thumbs/06.jpg",code:"/nougat/pages/06.mmd",markdown:null,pageNum:4,name:"Scanned Master Thesis (2023) - Wallner"},{img:"/nougat/pages/07.jpg",thumb:"/nougat/pages/thumbs/07.jpg",code:"/nougat/pages/07.mmd",markdown:null,pageNum:6,name:"Hierarchical Neural Story Generation (2018) - Fan et al.",link:"https://aclanthology.org/P18-1082.pdf"},{img:"/nougat/pages/08.jpg",thumb:"/nougat/pages/thumbs/08.jpg",code:"/nougat/pages/08.mmd",markdown:null,pageNum:6,name:"Cycle-Consistency for Robust Visual Question Answering (2019) - Shah et al.",link:"https://arxiv.org/pdf/1902.05660.pdf"}]}},methods:{loadPaper:function(){var t=this;return Object(H.a)(regeneratorRuntime.mark((function e(){return regeneratorRuntime.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:if(t.loading=!0,t.paper.markdown){e.next=5;break}return e.next=4,t.getMMD(t.paper.code);case 4:t.paper.markdown=e.sent;case 5:t.loading=!1;case 6:case"end":return e.stop()}}),e)})))()},getMMD:function(t){return Object(H.a)(regeneratorRuntime.mark((function e(){return regeneratorRuntime.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return e.next=2,fetch(t);case 2:return e.next=4,e.sent.text();case 4:return e.abrupt("return",e.sent);case 5:case"end":return e.stop()}}),e)})))()}},computed:{paper:function(){return this.papers[this.page-1]}},mounted:function(){this.loadPaper()}}),J=(n(555),n(74)),component=Object(J.a)(K,(function(){var t=this,e=t._self._c;return e(O.a,[e(r.b,[t._v(" Example Pages ")]),t._v(" "),e(r.a,[e(F,{staticClass:"mb-2",attrs:{length:t.papers.length,circle:""},on:{input:t.loadPaper},model:{value:t.page,callback:function(e){t.page=e},expression:"page"}}),t._v(" "),t.paper?e("div",[t.$vuetify.breakpoint.mobile?e("div",[e($,{staticClass:"mx-auto",staticStyle:{border:"solid #e0e0e0 1px"},attrs:{contain:"","max-height":"800","max-width":"565",alt:"Document Page",src:t.paper.img,"lazy-src":t.paper.thumb}}),t._v(" "),e(G.a,{staticClass:"overflow-y-auto",attrs:{"max-height":"800","min-height":"800",height:"800"}},[e("MarkdownDisplay",{attrs:{loading:t.loading,source:t.paper.markdown}})],1)],1):e(V.a,[e(S,{attrs:{cols:"6"}},[e($,{staticClass:"mx-auto",staticStyle:{border:"solid #e0e0e0 1px"},attrs:{contain:"","max-height":"707","max-width":"500",alt:"Document Page",src:t.paper.img,"lazy-src":t.paper.thumb}})],1),t._v(" "),e(S,{attrs:{cols:"6"}},[e(G.a,{staticClass:"overflow-y-auto",attrs:{"max-height":"707","min-height":"707",height:"707"}},[e("MarkdownDisplay",{attrs:{loading:t.loading,source:t.paper.markdown}})],1)],1)],1)],1):t._e(),t._v(" "),e("div",{staticClass:"mt-2 center-text"},[e("b",[t._v(t._s(t.paper.name))]),e("br"),t._v(" "),t.paper.link?e("span",[e(k.a,{staticClass:"mb-1 pr-1",attrs:{small:""}},[t._v("mdi-open-in-new")]),e("a",{attrs:{href:t.paper.link,target:"_blank",rel:"noopener noreferrer"}},[t._v("source")]),t._v(", page "+t._s(t.paper.pageNum))],1):t._e()])],1)],1)}),[],!1,null,null,null);e.default=component.exports;installComponents(component,{MarkdownDisplay:n(378).default})}}]);