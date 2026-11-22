// Caches only the static app shell (this page, manifest, icons) so it loads
// instantly and can open even with a flaky connection. Deliberately never
// touches the live model API: that's a different origin (Render), and the
// same-origin check below already excludes it, but the point stands even if
// this file is ever reused elsewhere — recommendations must always be live,
// never served stale from a cache.
const CACHE_NAME = "find-a-facility-v1";
const SHELL_ASSETS = ["./", "./index.html", "./manifest.json", "./icons/icon-192.png", "./icons/icon-512.png"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_ASSETS)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))))
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== "GET" || url.origin !== self.location.origin) {
    return; // not a same-origin GET for the shell: let the browser handle it normally
  }
  event.respondWith(caches.match(event.request).then((cached) => cached || fetch(event.request)));
});
