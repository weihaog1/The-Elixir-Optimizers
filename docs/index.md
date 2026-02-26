---
layout: default
title: Home
---

<section class="hero split-layout">
  <div class="hero-text reveal">
    <p class="elixir-badge elixir-badge--gold">CS 175 -- Winter 2026</p>
    <h1 class="hero-title site-title-shimmer">The Elixir<br>Optimizers</h1>
    <p class="hero-subtitle" style="font-size: 0.95rem; color: rgba(255,255,255,0.7); margin-bottom: 0.5rem; font-style: italic;">
      Named after the elixir resource in Clash Royale -- we optimize how every drop is spent.
    </p>
    <p class="hero-tagline">
      A reinforcement learning agent that plays Clash Royale
      through screen capture, computer vision, imitation learning,
      and reinforcement learning.
    </p>
    <div class="hero-actions">
      <a href="{{ 'proposal.html' | relative_url }}" class="btn-royal">Read Proposal</a>
      <a href="https://github.com/weihaog1/The-Elixir-Optimizers" class="btn-royal btn-royal--outline">Source Code</a>
    </div>
  </div>
  <div class="hero-visual reveal reveal-delay-1">
    <div class="hero-card-fan">
      <img src="{{ 'images/royal-recruits-card.png' | relative_url }}" alt="Royal Recruits" class="fan-card fan-card-1">
      <img src="{{ 'images/flying-machine-card.png' | relative_url }}" alt="Flying Machine" class="fan-card fan-card-2">
      <img src="{{ 'images/royal-hogs-card.png' | relative_url }}" alt="Royal Hogs" class="fan-card fan-card-3">
      <img src="{{ 'images/zappies-card.png' | relative_url }}" alt="Zappies" class="fan-card fan-card-4">
    </div>
  </div>
</section>

<hr class="divider">

<section class="reveal">
  <h2>How It Works</h2>
  <div class="steps-grid">
    <div class="arena-card reveal reveal-delay-1">
      <h3>Perception</h3>
      <p>
        YOLOv8s object detection identifies troops, spells, and buildings
        on the arena. OCR extracts elixir, timer, and tower HP values.
        A MiniResNet classifier recognizes the 4 cards in hand.
      </p>
    </div>
    <div class="arena-card reveal reveal-delay-2">
      <h3>Understanding</h3>
      <p>
        A state builder combines all detections into a structured
        game state: an 18x32 arena grid with unit channels,
        plus a vector of scalar features.
      </p>
    </div>
    <div class="arena-card reveal reveal-delay-3">
      <h3>Decision</h3>
      <p>
        A policy network trained via behavior cloning, then fine-tuned
        with PPO, selects which card to play and where to place it
        on the arena.
      </p>
    </div>
  </div>
</section>

<hr class="divider">

<section class="reveal">
  <h2>Pipeline Architecture</h2>
  <div class="arena-card" style="padding: 1rem; overflow-x: auto;">
    <img src="{{ 'images/pipeline-architecture.svg' | relative_url }}" alt="Pipeline Architecture Diagram" style="width: 100%; max-width: 800px; display: block; margin: 0 auto;">
  </div>
</section>

<hr class="divider">

<section class="reveal">
  <h2>Reports</h2>
  <ul class="report-list">
    <li>
      <a href="{{ 'proposal.html' | relative_url }}">Proposal</a>
      <span class="elixir-badge">Complete</span>
    </li>
    <li>
      <a href="{{ 'status.html' | relative_url }}">Status Report</a>
      <span class="elixir-badge">Complete</span>
    </li>
    <li>
      <a href="{{ 'final.html' | relative_url }}">Final Report</a>
      <span class="elixir-badge elixir-badge--crimson">Upcoming</span>
    </li>
  </ul>
</section>
