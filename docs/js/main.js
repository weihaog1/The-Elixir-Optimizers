(function () {
  'use strict';

  // --- Mobile nav toggle ---
  var toggle = document.getElementById('nav-toggle');
  var links = document.getElementById('nav-links');

  if (toggle && links) {
    toggle.addEventListener('click', function () {
      links.classList.toggle('open');
      toggle.classList.toggle('open');
    });

    // Close menu when a link is clicked
    var navAnchors = links.querySelectorAll('a');
    for (var i = 0; i < navAnchors.length; i++) {
      navAnchors[i].addEventListener('click', function () {
        links.classList.remove('open');
        toggle.classList.remove('open');
      });
    }
  }

  // --- Nav scroll state ---
  var nav = document.getElementById('site-nav');

  function onScroll() {
    if (!nav) return;
    if (window.scrollY > 40) {
      nav.classList.add('scrolled');
    } else {
      nav.classList.remove('scrolled');
    }
  }

  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  // --- Scroll reveal (IntersectionObserver) ---
  var reveals = document.querySelectorAll('.reveal');

  if ('IntersectionObserver' in window && reveals.length > 0) {
    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add('revealed');
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: '0px 0px -40px 0px' }
    );

    for (var j = 0; j < reveals.length; j++) {
      observer.observe(reveals[j]);
    }
  } else {
    // Fallback: show everything immediately
    for (var k = 0; k < reveals.length; k++) {
      reveals[k].classList.add('revealed');
    }
  }
})();
