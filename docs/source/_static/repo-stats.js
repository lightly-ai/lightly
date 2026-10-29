// Star and fork counts for the repository card in the sidebar
// (_templates/sidebar/github.html). The counts come from the GitHub API and the
// result is cached for the browser session, so browsing the docs costs one
// request. A failed request, for example when the API rate limit is reached, is
// cached too: the card then shows only the repository name until the session ends.
document.addEventListener("DOMContentLoaded", async () => {
  const card = document.querySelector(".ls-repo[data-repo]");
  if (!card) {
    return;
  }
  const repo = card.dataset.repo;
  const cacheKey = `ls-repo-stats:${repo}`;

  let stats;
  try {
    stats = JSON.parse(sessionStorage.getItem(cacheKey));
  } catch {
    // Storage can be unavailable, for example in some private windows.
  }
  if (!stats) {
    stats = await fetchStats(repo);
    try {
      sessionStorage.setItem(cacheKey, JSON.stringify(stats));
    } catch {
      // Not cached; the next page asks the API again.
    }
  }
  if (stats.failed) {
    return;
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

// Returns {stars, forks}, or {failed: true} if the API gives no usable counts.
async function fetchStats(repo) {
  try {
    const response = await fetch(`https://api.github.com/repos/${repo}`);
    if (!response.ok) {
      return { failed: true };
    }
    const data = await response.json();
    const stars = data.stargazers_count;
    const forks = data.forks_count;
    if (!Number.isFinite(stars) || !Number.isFinite(forks)) {
      return { failed: true };
    }
    return { stars, forks };
  } catch {
    return { failed: true };
  }
}
