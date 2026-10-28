// Family filter on the Models page. Each button carries data-family; each card
// column carries an ls-family-<family> class. Clicking a button hides the columns
// of the other families. The counts are read from the cards, so adding a card
// never leaves a button count out of date.
document.addEventListener("DOMContentLoaded", () => {
  const buttons = document.querySelectorAll(".ls-filter-btn");
  const items = document.querySelectorAll(".ls-filter-target .ls-fam-item");
  if (buttons.length === 0 || items.length === 0) {
    return;
  }

  const matches = (item, family) =>
    family === "all" || item.classList.contains(`ls-family-${family}`);

  buttons.forEach((button) => {
    const family = button.dataset.family;
    const badge = document.createElement("span");
    badge.className = "ls-filter-count";
    badge.textContent = [...items].filter((item) => matches(item, family)).length;
    button.append(badge);

    button.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.toggle("is-active", b === button));
      items.forEach((item) => {
        item.classList.toggle("is-hidden", !matches(item, family));
      });
    });
  });
});
