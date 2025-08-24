function startCelebration() {
  const name = document.getElementById("nameInput").value;
  const message = document.getElementById("message");
  const celebration = document.getElementById("celebration");

  if(name.trim() === "") {
    message.innerHTML = "Please enter a name to start the celebration!";
    celebration.style.display = "none";
  } else {
    celebration.style.display = "block";
    message.innerHTML = `🎈 Happy Birthday, ${name}! ✨<br> wish you happy birthday ! 🎂💖`;
    launchConfetti();
    launchBalloons();
  }
}

function launchConfetti() {
  for(let i=0; i<50; i++) {
    let confetti = document.createElement("div");
    confetti.classList.add("confetti");
    confetti.style.left = Math.random() * 100 + "vw";
    confetti.style.backgroundColor = getRandomColor();
    confetti.style.animationDuration = Math.random() * 3 + 2 + "s";
    document.body.appendChild(confetti);

    setTimeout(() => {
      confetti.remove();
    }, 5000);
  }
}

function launchBalloons() {
  for(let i=0; i<20; i++) {
    let balloon = document.createElement("div");
    balloon.classList.add("balloon");
    balloon.style.left = Math.random() * 100 + "vw";
    balloon.style.animationDuration = Math.random() * 4 + 4 + "s";
    balloon.innerHTML = "🎈";
    document.body.appendChild(balloon);

    setTimeout(() => {
      balloon.remove();
    }, 6000);
  }
}

function getRandomColor() {
  const colors = ["#ff0", "#f0f", "#0ff", "#f00", "#0f0", "#00f", "#ff69b4", "#ffa500"];
  return colors[Math.floor(Math.random() * colors.length)];
}
