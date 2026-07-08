(function () {
  const mount = document.getElementById("fitto-chat-widget");
  if (!mount) return;

  const fittoSrc = mount.dataset.fittoSrc || "";
  const fittoHeaderSrc = "/static/images/fitto-widget-header.png";
  const historyLimit = 6;
  const chatHistory = [];
  let isLoading = false;

  mount.innerHTML = `
    <section class="fitto-chat" aria-label="Чат с Фитто">
      <div class="fitto-chat__panel" role="dialog" aria-label="Фитто — AI-гид по саду решений" aria-hidden="true">
        <header class="fitto-chat__header">
          <span class="fitto-chat__mini-avatar" aria-hidden="true">
            <img src="${fittoHeaderSrc}" alt="">
          </span>
          <div>
            <p class="fitto-chat__title">Фитто</p>
            <p class="fitto-chat__subtitle">AI-гид по саду решений</p>
          </div>
          <button class="fitto-chat__close" type="button" aria-label="Закрыть чат">×</button>
        </header>
        <div class="fitto-chat__messages" aria-live="polite"></div>
        <form class="fitto-chat__form">
          <input class="fitto-chat__input" type="text" placeholder="Спросите Фитто..." autocomplete="off" aria-label="Сообщение для Фитто">
          <button class="fitto-chat__send" type="submit" aria-label="Отправить сообщение">→</button>
        </form>
      </div>
      <button class="fitto-chat__toggle" type="button" aria-label="Открыть чат с Фитто" aria-expanded="false">
        <img class="fitto-chat__avatar" src="${fittoSrc}" alt="">
        <span class="fitto-chat__hint" aria-hidden="true">Спросить Фитто</span>
        <span class="fitto-chat__status" aria-hidden="true"></span>
      </button>
    </section>
  `;

  const widget = mount.querySelector(".fitto-chat");
  const panel = mount.querySelector(".fitto-chat__panel");
  const toggle = mount.querySelector(".fitto-chat__toggle");
  const closeButton = mount.querySelector(".fitto-chat__close");
  const messages = mount.querySelector(".fitto-chat__messages");
  const form = mount.querySelector(".fitto-chat__form");
  const input = mount.querySelector(".fitto-chat__input");
  const sendButton = mount.querySelector(".fitto-chat__send");
  let attentionTimer = null;

  function isAllowedInternalPath(path) {
    return /^\/(?:(?:projects|articles)(?:\/[A-Za-z0-9_-]+)?|about|contacts)$/.test(path);
  }

  function appendLink(parent, href, label) {
    const link = document.createElement("a");
    link.className = "fitto-chat-link";
    link.href = href;
    link.textContent = label;
    parent.appendChild(link);
  }

  function appendBareLinks(parent, text) {
    const urlPattern = /`?(\/(?:(?:projects|articles)(?:\/[A-Za-z0-9_-]+)?|about|contacts))`?/g;
    let cursor = 0;
    let match;

    while ((match = urlPattern.exec(text)) !== null) {
      const [raw, href] = match;
      if (!isAllowedInternalPath(href)) continue;
      parent.appendChild(document.createTextNode(text.slice(cursor, match.index)));
      appendLink(parent, href, href);
      cursor = match.index + raw.length;
    }

    parent.appendChild(document.createTextNode(text.slice(cursor)));
  }

  function appendLinkedText(parent, text) {
    const markdownPattern = /\[([^\]]+)\]\((\/(?:(?:projects|articles)(?:\/[A-Za-z0-9_-]+)?|about|contacts))\)/g;
    let cursor = 0;
    let match;

    while ((match = markdownPattern.exec(text)) !== null) {
      const [raw, label, href] = match;
      if (!isAllowedInternalPath(href)) continue;
      appendBareLinks(parent, text.slice(cursor, match.index));
      appendLink(parent, href, label);
      cursor = match.index + raw.length;
    }

    appendBareLinks(parent, text.slice(cursor));
  }

  function addMessage(text, type) {
    const message = document.createElement("div");
    message.className = `fitto-chat__message fitto-chat__message--${type}`;
    if (type === "bot") {
      appendLinkedText(message, text);
    } else {
      message.textContent = text;
    }

    messages.appendChild(message);
    messages.scrollTop = messages.scrollHeight;
    return message;
  }

  function rememberMessage(role, content) {
    const cleanContent = String(content || "").trim();
    if (!cleanContent) return;

    chatHistory.push({
      role,
      content: cleanContent,
    });

    if (chatHistory.length > historyLimit) {
      chatHistory.splice(0, chatHistory.length - historyLimit);
    }
  }

  function setOpen(isOpen) {
    widget.classList.toggle("is-open", isOpen);
    widget.classList.remove("is-attention");
    panel.setAttribute("aria-hidden", String(!isOpen));
    toggle.setAttribute("aria-expanded", String(isOpen));
    toggle.setAttribute("aria-label", isOpen ? "Закрыть чат с Фитто" : "Открыть чат с Фитто");
    if (isOpen) {
      window.setTimeout(() => input.focus(), 120);
    }
  }

  function setLoading(loading) {
    isLoading = loading;
    sendButton.disabled = loading;
    input.disabled = loading;
  }

  async function sendMessage(userMessage) {
    const historyForRequest = chatHistory.slice(-historyLimit);
    addMessage(userMessage, "user");
    rememberMessage("user", userMessage);
    setLoading(true);
    const thinking = addMessage("Фитто думает...", "bot");

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: userMessage,
          history: historyForRequest,
        }),
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.error || "Request failed");
      }

      thinking.remove();
      const answer = data.answer || "Я пока не нашёл, что ответить.";
      addMessage(answer, "bot");
      rememberMessage("assistant", answer);
    } catch (error) {
      thinking.remove();
      addMessage("Кажется, у меня запутались провода. Попробуйте ещё раз чуть позже.", "bot");
    } finally {
      setLoading(false);
      input.focus();
    }
  }

  addMessage(
    "Привет! Я Фитто — робот-садовник этого сайта. Могу подсказать про проекты Александры, статьи и термины вроде RAG, LoRA или embeddings.",
    "bot"
  );

  widget.classList.add("is-attention");
  attentionTimer = window.setTimeout(() => {
    widget.classList.remove("is-attention");
  }, 5200);

  toggle.addEventListener("click", () => {
    if (attentionTimer) {
      window.clearTimeout(attentionTimer);
      attentionTimer = null;
    }
    setOpen(!widget.classList.contains("is-open"));
  });

  closeButton.addEventListener("click", () => {
    setOpen(false);
    toggle.focus();
  });

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (isLoading) return;
    const userMessage = input.value.trim();
    if (!userMessage) return;
    input.value = "";
    sendMessage(userMessage);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && widget.classList.contains("is-open")) {
      setOpen(false);
      toggle.focus();
    }
  });
})();
