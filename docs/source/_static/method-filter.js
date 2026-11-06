// Family filter on the start page and the Models page. Each button carries
// data-family; each family item (a card column on the Models page, a group of
// chips on the start page) carries an ls-family-<family> class. Clicking a
// button hides the items of the other families. A button's count is the number
// of method links in its items, so adding a method never leaves it out of date.
// A button whose family has no methods is disabled instead of emptying the list.
document.addEventListener("DOMContentLoaded", () => {
  const buttons = document.querySelectorAll(".ls-filter-btn");
  const items = [...document.querySelectorAll(".ls-filter-target .ls-fam-item")];
  if (buttons.length === 0 || items.length === 0) {
    return;
  }

  const matches = (item, family) =>
    family === "all" || item.classList.contains(`ls-family-${family}`);

  buttons.forEach((button) => {
    const family = button.dataset.family;
    const count = items
      .filter((item) => matches(item, family))
      .reduce((n, item) => n + item.querySelectorAll("a.reference").length, 0);

    // Read as "Contrastive, 5 methods" instead of "Contrastive5".
    const noun = count === 1 ? "method" : "methods";
    button.setAttribute("aria-label", `${button.textContent.trim()}, ${count} ${noun}`);
    const badge = document.createElement("span");
    badge.className = "ls-filter-count";
    badge.textContent = count;
    button.append(badge);
    button.setAttribute("aria-pressed", button.classList.contains("is-active"));
    button.disabled = count === 0;

    button.addEventListener("click", () => {
      buttons.forEach((b) => {
        b.classList.toggle("is-active", b === button);
        b.setAttribute("aria-pressed", b === button);
      });
      items.forEach((item) => {
        item.classList.toggle("is-hidden", !matches(item, family));
      });
    });
  });
});
