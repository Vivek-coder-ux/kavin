const API_KEY = "b8dc880d09ab411b8282a622a285d672";
const url = "https://newsapi.org/v2/everything?q=";

window.addEventListener('load', () => fetchNews("India"));

document.getElementById("ipl").addEventListener("click", () => {
  fetchNews("Cricket");
});
document.getElementById("finance").addEventListener("click", () => {
  fetchNews("Finance");
});
document.getElementById("politics").addEventListener("click", () => {
  fetchNews("Politics");
});

document.getElementById("search-button").addEventListener("click", () => {
  const query = document.getElementById("search-text").value;
  if (query) fetchNews(query);
});

async function fetchNews(query) {
  try {
    const res = await fetch(`${url}${query}&apiKey=${API_KEY}`);
    const data = await res.json();
    bindData(data.articles);
  } catch (error) {
    console.error("Error fetching news:", error);
    alert("There was an error fetching the news. Please try again later.");
  }
}

function bindData(articles) {
  const cardsContainer = document.getElementById('cards-container');
  const newsCardTemplate = document.getElementById('template-news-card');

  cardsContainer.innerHTML = '';

  if (!articles || articles.length === 0) {
    cardsContainer.innerHTML = '<p>No articles found. Try a different search.</p>';
    return;
  }

  articles.forEach(article => {
    if (!article.urlToImage) return;

    const cardClone = newsCardTemplate.content.cloneNode(true);

    const newsImg = cardClone.querySelector(".news-img");
    const newsTitle = cardClone.querySelector(".news-title");
    const newsSource = cardClone.querySelector(".news-source");
    const newsDesc = cardClone.querySelector(".news-desc");

    newsImg.src = article.urlToImage || "https://via.placeholder.com/400x200";
    newsTitle.textContent = article.title || "No Title Available";
    newsDesc.textContent = article.description || "No description available.";

    const date = new Date(article.publishedAt).toLocaleString("en-US", {
      timeZone: "Asia/Kolkata"
    });

    newsSource.textContent = `${article.source.name} • ${date}`;

    cardClone.querySelector(".card").addEventListener("click", () => {
      window.open(article.url, "_blank");
    });

    cardsContainer.appendChild(cardClone);
  });
}

