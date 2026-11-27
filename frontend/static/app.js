// Show the loading indicator while a search is in flight.
(function () {
  var form = document.getElementById("search-form");
  var loading = document.getElementById("loading");
  var button = document.getElementById("search-button");
  if (!form || !loading || !button) return;

  form.addEventListener("submit", function () {
    loading.hidden = false;
    button.disabled = true;
    button.textContent = "Searching…";
  });

  // Reset if the user comes back to this page via the browser's back button.
  window.addEventListener("pageshow", function () {
    loading.hidden = true;
    button.disabled = false;
    button.textContent = "Search";
  });
})();
