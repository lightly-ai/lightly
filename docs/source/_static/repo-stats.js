// Star and fork counts for the repository card in the sidebar
// (_templates/sidebar/github.html). The counts come from the GitHub API and are
// cached for the browser session, so browsing the docs costs one request. If the
// request fails, for example when the API rate limit is reached, the card keeps
// showing only the repository name.
document.addEventListener("DOMContentLoaded", async () => {
  const card = document.querySelector(".ls-repo[data-repo]");
  if (!card) {
    return;
  }
  const repo = card.dataset.repo;
  const cacheKey = `ls-repo-stats:${repo}`;

  let stats = null;
  try {
    stats = JSON.parse(sessionStorage.getItem(cacheKey));
  } catch {
    // Storage can be unavailable, for example in some private windows.
  }
  if (!stats) {
    try {
      const response = await fetch(`https://api.github.com/repos/${repo}`);
      if (!response.ok) {
        return;
      }
      const data = await response.json();
      stats = { stars: data.stargazers_count, forks: data.forks_count };
    } catch {
      return;
    }
    try {
      sessionStorage.setItem(cacheKey, JSON.stringify(stats));
    } catch {
      // Not cached; the next page fetches the counts again.
    }
  }

  // 3812 -> "3.8k", as in the Material for MkDocs source link.
  const format = new Intl.NumberFormat("en", {
    notation: "compact",
    maximumFractionDigits: 1,
  });
  card.querySelector("[data-repo-stars]").textContent = format
    .format(stats.stars)
    .toLowerCase();
  card.querySelector("[data-repo-forks]").textContent = format
    .format(stats.forks)
    .toLowerCase();
  card.querySelector(".ls-repo-facts").hidden = false;
});
