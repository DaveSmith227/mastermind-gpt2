function setUpSpecialNavs(){$(".navbar-toggler").click(function(a){var b=$(this).closest("nav"),c=b.find("ul.site-navigation"),d=c.clone();c.parent().is(".fullscreen-nav, .sidebar-nav")&&((a.stopPropagation(),c.parent().addClass("nav-special"),$(this).hasClass("selected-nav"))?($(".blocsapp-special-menu blocsnav").removeClass("open"),$(".selected-nav").removeClass("selected-nav"),setTimeout(function(){$(".blocsapp-special-menu").remove();$("body").removeClass("lock-scroll");$(".nav-special").removeClass("nav-special")},
300)):($(this).addClass("selected-nav"),a=b.attr("class").replace("navbar","").replace("row",""),c=c.parent().attr("class").replace("navbar-collapse","").replace("collapse",""),$(".content-tint").length=-1,$("body").append('<div class="content-tint"></div>'),d.insertBefore(".page-container").wrap('<div class="blocsapp-special-menu '+a+'"><blocsnav class="'+c+'">'),$("blocsnav").prepend('<a class="close-special-menu animated fadeIn" style="animation-delay:0.5s;"><div class="close-icon"></div></a>'),
function(){var a="fadeInRight",b=0,c=60;$(".blocsapp-special-menu blocsnav").hasClass("fullscreen-nav")?(a="fadeIn",c=100):$(".blocsapp-special-menu").hasClass("nav-invert")&&(a="fadeInLeft");$(".blocsapp-special-menu blocsnav li").each(function(){$(this).parent().hasClass("dropdown-menu")?$(this).addClass("animated fadeIn"):(b+=c,$(this).attr("style","animation-delay:"+b+"ms").addClass("animated "+a))})}(),setTimeout(function(){$(".blocsapp-special-menu blocsnav").addClass("open");$(".content-tint").addClass("on");
$("body").addClass("lock-scroll")},10)))});$("body").on("mousedown touchstart",".content-tint, .close-special-menu",function(a){$(".content-tint").removeClass("on");$(".selected-nav").click();setTimeout(function(){$(".content-tint").remove()},10)}).on("click",".blocsapp-special-menu a",function(a){$(a.target).closest(".dropdown-toggle").length||$(".close-special-menu").mousedown()})}
function extraNavFuncs(){$(".site-navigation a").click(function(a){$(a.target).closest(".dropdown-toggle").length||$(".navbar-collapse").collapse("hide")});$("a.dropdown-toggle").click(function(a){$(this).parent().addClass("target-open-menu");$(this).closest(".dropdown-menu").find(".dropdown.open").each(function(a){$(this).hasClass("target-open-menu")||$(this).removeClass("open")});$(".target-open-menu").removeClass("target-open-menu")});$(".dropdown-menu a.dropdown-toggle").on("click",function(a){a=
$(this);var b=$(this).offsetParent(".dropdown-menu");return $(this).next().hasClass("show")||$(this).parents(".dropdown-menu").first().find(".show").removeClass("show"),$(this).next(".dropdown-menu").toggleClass("show"),$(this).parent("li").toggleClass("show"),$(this).parents("li.nav-item.dropdown.show").on("hidden.bs.dropdown",function(a){$(".dropdown-menu .show").removeClass("show")}),b.parent().hasClass("navbar-nav")||a.next().css({top:a[0].offsetTop,left:b.outerWidth()-4}),!1})}
function scrollToTarget(a,b){var c="slow";0==a?a=$(b).closest(".bloc").height():1==a?a=0:2==a?a=$(document).height():(a=$(a).offset().top,$(".sticky-nav").length&&(a-=$(".sticky-nav").outerHeight()));$(b).is("[data-scroll-speed]")&&(c=$(b).attr("data-scroll-speed"),parseInt(c)&&(c=parseInt(c)));$("html,body").animate({scrollTop:a},c);$(".navbar-collapse").collapse("hide")}
function animateWhenVisible(){hideAll();inViewCheck();$(window).scroll(function(){inViewCheck();scrollToTopView();stickyNavToggle()})}function setUpDropdownSubs(){$("ul.dropdown-menu [data-toggle=dropdown]").on("click",function(a){a.preventDefault();a.stopPropagation();$(this).parent().siblings().removeClass("open");$(this).parent().toggleClass("open");a=$(this).parent().children(".dropdown-menu");a.offset().left+a.width()>$(window).width()&&a.addClass("dropmenu-flow-right")})}
function stickyNavToggle(){var a=0,b="sticky";$(".sticky-nav").hasClass("fill-bloc-top-edge")&&(a=$(".fill-bloc-top-edge.sticky-nav").parent().css("background-color"),"rgba(0, 0, 0, 0)"==a&&(a="#FFFFFF"),$(".sticky-nav").css("background",a),a=$(".sticky-nav").height(),b="sticky animated fadeInDown");$(window).scrollTop()>a?($(".sticky-nav").addClass(b),"sticky"==b&&$(".page-container").css("padding-top",$(".sticky-nav").height())):($(".sticky-nav").removeClass(b).removeAttr("style"),$(".page-container").removeAttr("style"))}
function hideAll(){$(".animated").each(function(a){$(this).closest(".hero").length||$(this).removeClass("animated").addClass("hideMe")})}
function inViewCheck(){$($(".hideMe").get().reverse()).each(function(a){var b=jQuery(this);a=b.offset().top+b.height();var c=$(window).scrollTop()+$(window).height();if(b.height()>$(window).height()&&(a=b.offset().top),a<c){var d=b.attr("class").replace("hideMe","animated");b.css("visibility","hidden").removeAttr("class");setTimeout(function(){b.attr("class",d).css("visibility","visible")},.01);b.on("webkitAnimationEnd mozAnimationEnd oAnimationEnd animationEnd",function(a){$(this).removeClass($(this).attr("data-appear-anim-style"))})}})}
function scrollToTopView(){$(window).scrollTop()>$(window).height()/3?$(".scrollToTop").hasClass("showScrollTop")||$(".scrollToTop").addClass("showScrollTop"):$(".scrollToTop").removeClass("showScrollTop")}
function setUpVisibilityToggle(){$(document).on("click","[data-toggle-visibility]",function(a){function b(a){a.is("img")?a.toggle():a.is(".row, .bloc-group")?a.toggleClass("d-flex"):a.slideToggle()}a.preventDefault();a=$(this).attr("data-toggle-visibility");if(-1!=a.indexOf(",")){var c=a.split(",");$.each(c,function(a){b($("#"+c[a]))})}else b($("#"+a))})}
function setUpLightBox(){window.targetLightbox;$(document).on("click","[data-lightbox]",function(a){a.preventDefault();targetLightbox=$(this);a=targetLightbox.attr("data-lightbox");var b=targetLightbox.attr("data-autoplay"),c='<p class="lightbox-caption">'+targetLightbox.attr("data-caption")+"</p>",d="no-gallery-set",e=targetLightbox.attr("data-frame");targetLightbox.attr("data-gallery-id")&&(d=targetLightbox.attr("data-gallery-id"));var f="";1==b&&(f="autoplay");b=$('<div id="lightbox-modal" class="modal fade"><div class="modal-dialog modal-dialog-centered modal-lg"><div class="modal-content '+
e+' blocs-lb-container"><button id="blocs-lightbox-close-btn" type="button" class="close-lightbox" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button><div class="modal-body"><a href="#" class="prev-lightbox" aria-label="prev"><span class="fa fa-chevron-left"></span></a><a href="#" class="next-lightbox" aria-label="next"><span class="fa fa-chevron-right"></span></a><img id="lightbox-image" class="img-fluid mx-auto d-block" src="'+a+'"><div id="lightbox-video-container" class="embed-responsive embed-responsive-16by9"><video controls '+
f+' class="embed-responsive-item"><source id="lightbox-video" src="'+a+'" type="video/mp4"></video></div>'+c+"</div></div></div></div>");$("body").append(b);"fullscreen-lb"==e&&($("#lightbox-modal").addClass("fullscreen-modal").append('<a class="close-full-screen-modal animated fadeIn" style="animation-delay:0.5s;" onclick="$(\'#lightbox-modal\').modal(\'hide\');"><div class="close-icon"></div></a>'),$("#blocs-lightbox-close-btn").remove());".mp4"==a.substring(a.length-4)?($("#lightbox-image, .lightbox-caption").removeClass("d-block").hide(),
$("#lightbox-video-container").show()):($("#lightbox-image,.lightbox-caption").addClass("d-block").show(),$("#lightbox-video-container").hide(),targetLightbox.attr("data-caption")||$(".lightbox-caption").removeClass("d-block").hide());$("#lightbox-modal").modal("show");"no-gallery-set"==d?(0==$("a[data-lightbox]").index(targetLightbox)&&$(".prev-lightbox").hide(),$("a[data-lightbox]").index(targetLightbox)==$("a[data-lightbox]").length-1&&$(".next-lightbox").hide()):(0==$('a[data-gallery-id="'+d+
'"]').index(targetLightbox)&&$(".prev-lightbox").hide(),$('a[data-gallery-id="'+d+'"]').index(targetLightbox)==$('a[data-gallery-id="'+d+'"]').length-1&&$(".next-lightbox").hide());addLightBoxSwipeSupport()}).on("hidden.bs.modal","#lightbox-modal",function(){$("#lightbox-modal").remove()});$(document).on("click",".next-lightbox, .prev-lightbox",function(a){a.preventDefault();a="no-gallery-set";var b=$("a[data-lightbox]").index(targetLightbox),c=$("a[data-lightbox]").eq(b+1);targetLightbox.attr("data-gallery-id")&&
(a=targetLightbox.attr("data-gallery-id"),b=$('a[data-gallery-id="'+a+'"]').index(targetLightbox),c=$('a[data-gallery-id="'+a+'"]').eq(b+1));$(this).hasClass("prev-lightbox")&&(c=$('a[data-gallery-id="'+a+'"]').eq(b-1),"no-gallery-set"==a&&(c=$("a[data-lightbox]").eq(b-1)));b=c.attr("data-lightbox");if(".mp4"==b.substring(b.length-4)){var d="";1==c.attr("data-autoplay")&&(d="autoplay");$("#lightbox-image, .lightbox-caption").removeClass("d-block").hide();$("#lightbox-video-container").show().html("<video controls "+
d+' class="embed-responsive-item"><source id="lightbox-video" src="'+b+'" type="video/mp4"></video>')}else $("#lightbox-image").attr("src",b).addClass("d-block").show(),$("#lightbox-video-container").hide(),$(".lightbox-caption").removeClass("d-block").hide(),c.attr("data-caption")&&$(".lightbox-caption").html(c.attr("data-caption")).show();targetLightbox=c;$(".next-lightbox, .prev-lightbox").hide();"no-gallery-set"==a?($("a[data-lightbox]").index(c)!=$("a[data-lightbox]").length-1&&$(".next-lightbox").show(),
0<$("a[data-lightbox]").index(c)&&$(".prev-lightbox").show()):($('a[data-gallery-id="'+a+'"]').index(c)!=$('a[data-gallery-id="'+a+'"]').length-1&&$(".next-lightbox").show(),0<$('a[data-gallery-id="'+a+'"]').index(c)&&$(".prev-lightbox").show())})}function addKeyBoardSupport(){$(window).keydown(function(a){37==a.which?$(".prev-lightbox").is(":visible")&&$(".prev-lightbox").click():39==a.which&&$(".next-lightbox").is(":visible")&&$(".next-lightbox").click()})}
function addLightBoxSwipeSupport(){$("#lightbox-image").length&&$("#lightbox-image").swipe({swipeLeft:function(a,b,c,d,e){$(".next-lightbox").is(":visible")&&$(".next-lightbox").click()},swipeRight:function(){$(".prev-lightbox").is(":visible")&&$(".prev-lightbox").click()},threshold:0})}
$(function(){extraNavFuncs();setUpSpecialNavs();setUpDropdownSubs();setUpLightBox();setUpVisibilityToggle();addKeyBoardSupport();$('a[onclick^="scrollToTarget"]').click(function(a){a.preventDefault()});$(".nav-item [data-active-page]").addClass($(".nav-item [data-active-page]").attr("data-active-page"));$('[data-toggle="tooltip"]').tooltip()});$(window).on("load",function(){animateWhenVisible();$("#page-loading-blocs-notifaction").remove()});