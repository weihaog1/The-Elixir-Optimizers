---
layout: default
title: Status
page_title: Status Report
page_subtitle: "Progress update on The Elixir Optimizers"
wide: true
---

<!-- Status page styles -->
<style>
  /* Content panels - readable cream background without distracting hover effects */
  .sp {
    background: var(--cr-cream);
    border: 2px solid var(--cr-brown-border);
    border-radius: var(--border-radius-lg);
    padding: var(--space-lg) var(--space-lg) var(--space-md);
    margin-bottom: var(--space-lg);
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }
  .sp p, .sp li { max-width: 72ch; }
  .sp p { margin-bottom: 0.85rem; }
  .sp h3 { margin-top: 0; }
  .sp h4 { margin-top: 1.5rem; }

  /* Metric highlight grid */
  .metric-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: var(--space-md);
    margin: var(--space-md) 0 var(--space-lg);
  }
  .metric-box {
    background: var(--cr-surface);
    border: 2px solid var(--cr-cream-dark);
    border-radius: var(--border-radius);
    padding: 0.75rem 1rem;
    text-align: center;
  }
  .metric-box .num {
    display: block;
    font-family: var(--font-heading);
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--cr-brown);
    line-height: 1.2;
  }
  .metric-box .label {
    display: block;
    font-size: 0.78rem;
    color: var(--cr-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 0.2rem;
  }
  .metric-box.gold { border-color: var(--cr-gold); }
  .metric-box.gold .num { color: var(--cr-gold-dim); }

  /* Two-column layout */
  .two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-lg);
    margin: var(--space-md) 0;
  }
  @media (max-width: 767px) {
    .two-col { grid-template-columns: 1fr; }
  }

  /* Steps / numbered pipeline modules */
  .step-card {
    background: var(--cr-surface);
    border: 2px solid var(--cr-cream-dark);
    border-radius: var(--border-radius);
    padding: 1rem 1.25rem;
    margin-bottom: var(--space-sm);
  }
  .step-card strong { color: var(--cr-blue-dark); }
  .step-card p { margin-bottom: 0.5rem; }
  .step-card p:last-child { margin-bottom: 0; }

  /* Challenge severity left-border accent */
  .challenge {
    border-left: 5px solid var(--cr-brown-border);
    padding-left: var(--space-md);
    margin-bottom: var(--space-lg);
  }
  .challenge.high { border-left-color: var(--cr-crimson); }
  .challenge.med  { border-left-color: var(--cr-gold); }
  .challenge.low  { border-left-color: var(--cr-green); }
  .challenge h4 { margin-top: 0; margin-bottom: 0.35rem; }
  .challenge p { margin-bottom: 0.5rem; }
  .challenge p:last-child { margin-bottom: 0; }
  .challenge ul { margin-bottom: 0; }

  /* Inline code in panels */
  .sp code {
    font-size: 0.85rem;
    padding: 0.1em 0.35em;
  }

  /* Section title on blue background */
  .section-title {
    color: #fff;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    font-family: var(--font-heading);
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    margin-bottom: var(--space-md);
  }

  /* Subsection title inside panels */
  .sp .sub-title {
    font-family: var(--font-heading);
    font-size: 1.1rem;
    color: var(--cr-brown);
    font-weight: 700;
    margin-top: 1.8rem;
    margin-bottom: 0.6rem;
    padding-bottom: 0.35rem;
    border-bottom: 2px solid var(--cr-cream-dark);
  }
  .sp .sub-title:first-child { margin-top: 0; }

  /* Compact table inside panels */
  .sp table { margin: 0.75rem 0 1rem; font-size: 0.9rem; }
  .sp th { font-size: 0.75rem; padding: 0.4rem 0.75rem; }
  .sp td { padding: 0.4rem 0.75rem; }
</style>

<!-- Video Summary -->
<section class="reveal">
<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; border-radius: 12px; border: 3px solid var(--cr-brown-border); box-shadow: 0 4px 16px rgba(0,0,0,0.15);">
  <iframe src="https://www.youtube.com/embed/X-sBAe_bAFs" title="The Elixir Optimizers - Status Report Video" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
</div>
</section>

<hr class="divider">


<!-- ============================================================ -->
<!-- PROJECT SUMMARY                                               -->
<!-- ============================================================ -->

<section class="reveal">
<h2 class="section-title">Project Summary</h2>
<div class="sp">

  <p>We are building an <strong>end-to-end reinforcement learning agent</strong> that plays Clash Royale, a real-time strategy card game, by interacting with the live game client running on Google Play Games (PC).</p>

  <p>The agent perceives the game state entirely through screen capture and computer vision:</p>
  <ul>
    <li><strong>YOLOv8</strong> object detection for identifying units, towers, and spells on the arena</li>
    <li><strong>PaddleOCR</strong> for reading numerical values (elixir, timer, tower HP)</li>
    <li><strong>MiniResNet</strong> classifier for recognizing cards in hand</li>
  </ul>

  <p>It then decides which card to play and where to place it, executing actions via automated mouse input through PyAutoGUI.</p>

  <p>Unlike approaches that rely on game memory injection or custom simulators, our system operates as an <strong>external observer</strong> -- the same way a human player does. This makes the problem significantly harder (we must solve perception before we can solve strategy), but the resulting agent interacts with the unmodified retail game.</p>

  <div class="metric-row">
    <div class="metric-box gold"><span class="num">0.804</span><span class="label">mAP50</span></div>
    <div class="metric-box"><span class="num">155</span><span class="label">Detection classes</span></div>
    <div class="metric-box"><span class="num">0.322</span><span class="label">BC F1 Score</span></div>
    <div class="metric-box"><span class="num">69%</span><span class="label">Action Recall</span></div>
    <div class="metric-box"><span class="num">23</span><span class="label">Live Sessions</span></div>
    <div class="metric-box"><span class="num">153</span><span class="label">Unit Tests</span></div>
  </div>

  <p>Our current prototype has a fully functional perception pipeline, a complete state representation and encoding system, a trained behavior cloning policy from expert demonstrations, and PPO fine-tuning currently in progress on live matches. The entire pipeline -- from screen capture through autonomous card placement -- has been tested end-to-end.</p>

</div>
</section>

<hr class="divider">


<!-- ============================================================ -->
<!-- APPROACH                                                      -->
<!-- ============================================================ -->

<section class="reveal">
<h2 class="section-title">Approach</h2>
<div class="sp">
  <p>Our system is a multi-stage pipeline that transforms raw screen pixels into card placement decisions. Below we describe each component, the technical decisions behind it, and the specific hyperparameters and architectures used.</p>
</div>
</section>


<!-- System Architecture -->

<section class="reveal">
<div class="sp">
  <h3>System Architecture</h3>

  <p>The full pipeline operates in two modes: <strong>data collection</strong> (for behavior cloning) and <strong>live inference</strong> (for autonomous play). Both share the same perception stack.</p>

  <div class="arena-card" style="padding: 1rem; overflow-x: auto;">
    <img src="{{ 'images/pipeline-architecture.svg' | relative_url }}" alt="Pipeline Architecture Diagram" style="width: 100%; max-width: 800px; display: block; margin: 0 auto;">
  </div>

  <p><strong>Data collection mode</strong> adds two parallel threads:</p>
  <ul>
    <li><strong>Thread A</strong> captures screenshots and builds GameState objects with timestamps</li>
    <li><strong>Thread B</strong> runs an OS-level mouse click logger that records the human player's card placements</li>
  </ul>
  <p>After a match, a DatasetBuilder merges these streams by timestamp into (observation, action) pairs stored as <code>.npz</code> files.</p>

</div>
</section>


<!-- Object Detection with YOLOv8 -->

<section class="reveal">
<div class="sp">
  <h3>Object Detection with YOLOv8</h3>

  <p>We use <strong>YOLOv8s</strong> (11.2M parameters) from Ultralytics as our primary detector. We chose the "small" variant over "nano" (3.2M params) after finding that nano lacked the capacity to distinguish between visually similar Clash Royale units (e.g., wizard vs. ice wizard, skeleton vs. skeleton army). YOLOv8s gave us 3.3x more parameters while still running at 15+ FPS on a single GPU.</p>

  <p>Our model detects <strong>155 distinct Clash Royale entities</strong> -- troops, buildings, spells, towers, UI elements (HP bars, level indicators), and background objects. We adopted the 155-class schema defined by the KataCR dataset, which covers all cards in the current game meta.</p>

  <h4 class="sub-title">Training on Synthetic Data</h4>

  <p>Inspired by KataCR's use of synthetic data, we built our own on-the-fly image generation pipeline to train exclusively on synthetically composited images. Each training batch is generated by our <code>Generator</code> module:</p>

  <ol>
    <li>A base <strong>arena background</strong> image is loaded (we use <code>background_index=15</code>, a stone/railroad arena that matches our gameplay footage).</li>
    <li><strong>Tower sprites</strong> are placed at fixed grid positions with randomized variants (intact, damaged, ruin). Generation ratios: 95% king tower intact, 75% queen tower (split evenly between queen, cannoneer, and dagger-duchess variants).</li>
    <li>Up to <strong>40 unit sprites</strong> are placed per image using an 18x32 occupancy grid with inverse-frequency weighted class sampling to address class imbalance. Ground and flying units use separate occupancy maps.</li>
    <li><strong>UI components</strong> (HP bars, level indicators) are attached to their parent units with configured positional offsets.</li>
    <li><strong>Augmentation</strong> is applied: color filters (2% each for red/blue/golden/white tints, 1% violet), scale/stretch perturbation, and a 50% chance of red-tinting the background.</li>
  </ol>

  <p>This produces <strong>20,000 unique synthetic images per epoch</strong>. The key advantage is unlimited training data with perfect annotations, at the cost of a domain gap between synthetic sprites and real in-game rendering.</p>

  <h4 class="sub-title">Training Configuration</h4>

  <table>
    <thead>
      <tr><th>Parameter</th><th>Value</th><th>Source</th></tr>
    </thead>
    <tbody>
      <tr><td>Base model</td><td>YOLOv8s (pretrained on COCO)</td><td>Ultralytics</td></tr>
      <tr><td>Image size</td><td>960x960</td><td>Tuned (640 missed small troops)</td></tr>
      <tr><td>Batch size</td><td>16</td><td>Hardware-limited</td></tr>
      <tr><td>Epochs</td><td>50</td><td>Early stopping patience=15</td></tr>
      <tr><td>Optimizer</td><td>SGD</td><td>Ultralytics default</td></tr>
      <tr><td>Learning rate</td><td>0.01 initial, linear decay, final factor 0.01</td><td>Ultralytics default</td></tr>
      <tr><td>Augmentation</td><td>degrees=5, scale=0.5, fliplr=0.5, erasing=0.4, HSV shifts</td><td>Tuned</td></tr>
      <tr><td>Mosaic/Mixup</td><td>Disabled</td><td>Generator handles compositing</td></tr>
      <tr><td>AMP</td><td>Enabled</td><td>Default</td></tr>
    </tbody>
  </table>

  <p><strong>Validation set:</strong> 1,388 human-annotated images from the KataCR published dataset. These are real Clash Royale screenshots with manually drawn bounding boxes, providing a ground-truth measure of how well our synthetic-trained model generalizes to real gameplay.</p>

  <div style="text-align: center; margin-top: 1rem;">
    <img src="{{ 'images/status/detection_demo.gif' | relative_url }}" alt="Real-time object detection on Clash Royale gameplay" style="width: 100%; max-width: 540px; border-radius: 8px; border: 2px solid var(--cr-brown-border);">
  </div>

</div>
</section>


<!-- Belonging Prediction -->

<section class="reveal">
<div class="sp">
  <h3>Belonging Prediction (Ally vs. Enemy Classification)</h3>

  <p>A critical challenge in Clash Royale is distinguishing ally units from enemy units -- visually identical troops differ only by a subtle blue/red tint. Inspired by how KataCR annotates belonging in its dataset labels, we extended the YOLOv8 architecture with a custom belonging prediction head.</p>

  <h4 class="sub-title">Architecture Modification</h4>

  <p>Our <code>CRDetectionModel</code> extends Ultralytics' <code>DetectionModel</code>. For a model with \(C\) object classes, the prediction tensor has \(C+1\) channels, where the last channel is a binary belonging signal (0 = ally, 1 = enemy). This is handled by our custom <code>CRDetectionLoss</code>:</p>

  <p>$$\mathcal{L}_{\text{belong}} = \text{BCE}(\hat{b}_i, b_i)$$</p>

  <p>where \(\hat{b}_i\) is the predicted belonging probability and \(b_i \in \{0, 1\}\) is the ground truth. This loss is folded into the existing classification loss -- the belonging channel is trained alongside the class channels via the same BCE loss, weighted by the Task-Aligned Assigner's alignment metric.</p>

  <h4 class="sub-title">Custom NMS</h4>

  <p>Our <code>non_max_suppression()</code> function produces <strong>7-column output tensors</strong> \((x_1, y_1, x_2, y_2, \text{conf}, \text{cls}, \text{belong})\) instead of the standard 6-column output. Belonging is thresholded at 0.5 to produce a binary label.</p>

  <h4 class="sub-title">Custom Sprites for Belonging Training</h4>

  <p>We created <strong>233 custom ally sprite cutouts</strong> for our specific deck (Royal Recruits Hog) using Quick-Seg (described below) and added them to a forked version of the KataCR dataset. This teaches the synthetic generator to produce ally-tinted sprites alongside the existing enemy-tinted ones.</p>

  <div class="step-card" style="margin-top: 1rem;">
    <p><strong>Current status:</strong> The belonging architecture and training infrastructure are complete. The current production <code>StateBuilder</code> still uses a Y-position heuristic as a fallback (units above 42% of screen height are classified as enemy). Switching to model-predicted belonging requires retraining with our custom ally sprites and validating on real gameplay, which is an immediate next step.</p>
  </div>

</div>
</section>


<!-- Quick-Seg -->

<section class="reveal">
<div class="sp">
  <h3>Quick-Seg: Custom Sprite Cutout Tool</h3>

  <p>A key bottleneck in our pipeline was creating custom sprite cutouts for training data generation. Existing annotation tools (LabelMe, CVAT, Roboflow) are designed for bounding boxes or full segmentation masks -- none provided a streamlined workflow for extracting transparent RGBA sprite cutouts from gameplay video, which is what our synthetic data generator requires.</p>

  <p>Alan developed <strong>Quick-Seg</strong>, a browser-based tool purpose-built for this workflow:</p>

  <ul>
    <li><strong>Flask + vanilla JavaScript</strong> application</li>
    <li>Loads gameplay video with frame-by-frame navigation</li>
    <li>Freehand lasso tool with add/subtract modes for precise sprite selection</li>
    <li>Exports cutouts as transparent RGBA PNGs organized by class directory</li>
  </ul>

  <p>The tool was used by team members to extract ally sprite cutouts from our gameplay footage, which are fed directly into the synthetic data generator's sprite pool for belonging-aware training. Quick-Seg is <a href="https://github.com/weihaog1/Quick-Seg">open-sourced on GitHub</a> as a standalone tool.</p>

  <p style="text-align: center; margin-top: 0.75rem;">
    <a href="https://github.com/weihaog1/Quick-Seg" style="display: inline-block; padding: 0.5rem 1.25rem; background: var(--cr-brown); color: var(--cr-cream); border-radius: var(--border-radius); font-weight: 600; text-decoration: none; font-size: 0.95rem; letter-spacing: 0.02em; transition: background 0.2s;">View Quick-Seg on GitHub &rarr;</a>
  </p>

  <div style="text-align: center; margin-top: 1rem;">
    <img src="{{ '/images/status/quick-seg-screeshot.png' | relative_url }}" alt="Screenshot of Quick-Seg tool interface" style="max-width: 100%; border-radius: var(--border-radius); border: 2px solid var(--cr-brown-border);">
  </div>

</div>
</section>


<!-- OCR Pipeline -->

<section class="reveal">
<div class="sp">
  <h3>OCR Pipeline</h3>

  <p>Numerical game values (elixir count, match timer, tower HP) cannot be detected by the object detection model -- they require optical character recognition. We use PaddleOCR with a specialized <code>GameTextExtractor</code> that handles region cropping, preprocessing, and value parsing across 9 screen regions.</p>

  <ul>
    <li><strong>Region-based extraction:</strong> Predefined crop regions (calibrated for 540x960 base resolution, scaled proportionally) isolate the timer, elixir bar, and 6 tower HP areas.</li>
    <li><strong>Preprocessing:</strong> CLAHE contrast enhancement with optional binary thresholding and inversion for small text.</li>
    <li><strong>Error-corrected parsing:</strong> Regex parsers handle common OCR mistakes (O to 0, I to 1), validate ranges (elixir 0-10, HP 0-10K, timer M:SS format), and detect overtime.</li>
  </ul>

</div>
</section>


<!-- Card Classification -->

<section class="reveal">
<div class="sp">
  <h3>Card Classification</h3>

  <p>Identifying which 4 cards are in the player's hand requires a separate classifier, since the card art in the hand tray is too small and stylized for the general object detector. We built a custom <strong>MiniResNet</strong> (~156K parameters) -- a 4-block residual network with an 8-class softmax head, one class per card in our deck.</p>

  <p>Rather than collecting hundreds of card screenshots, we train from just <strong>8 reference images</strong> (one per card) using heavy augmentation: <code>RandomAffine</code>, <code>ColorJitter</code>, <code>GaussianBlur</code>, <code>RandomErasing</code>, and a custom <code>apply_greyout()</code> that simulates the in-game greyed-out effect when a card costs more elixir than available. Greyout probability is tuned per card -- Royal Recruits (7 elixir) gets 70% since it is greyed out most of the time. Each epoch generates <strong>4,000 augmented samples</strong> (500 per class), trained with AdamW and CosineAnnealing.</p>

</div>
</section>


<!-- State Representation and Encoding -->

<section class="reveal">
<div class="sp">
  <h3>State Representation and Encoding</h3>

  <p>The <code>StateBuilder</code> orchestrates the full perception pipeline: it runs YOLOv8 detection, OCR extraction, and assembles the results into a <code>GameState</code> dataclass containing:</p>

  <ul>
    <li><strong>Units:</strong> list of detected units with class name, bounding box, confidence, and belonging (ally/enemy)</li>
    <li><strong>Towers:</strong> 6 tower slots (3 player, 3 enemy) with type, HP, and status</li>
    <li><strong>Elixir:</strong> current elixir count (0-10)</li>
    <li><strong>Timer:</strong> match time remaining in seconds</li>
    <li><strong>Cards:</strong> 4 card slots with class name and elixir cost (when card classifier is wired in)</li>
  </ul>

  <p>The <code>StateEncoder</code> then converts this variable-length <code>GameState</code> into fixed-shape tensors suitable for Stable Baselines 3.</p>

  <h4 class="sub-title">Arena Observation -- (32 x 18 x 7) float32</h4>

  <table>
    <thead>
      <tr><th>Channel</th><th>Content</th><th>Encoding</th></tr>
    </thead>
    <tbody>
      <tr><td>0</td><td>Ally ground units</td><td>Count of ally ground troops in cell</td></tr>
      <tr><td>1</td><td>Ally flying units</td><td>Count of ally flying troops in cell</td></tr>
      <tr><td>2</td><td>Enemy ground units</td><td>Count of enemy ground troops in cell</td></tr>
      <tr><td>3</td><td>Enemy flying units</td><td>Count of enemy flying troops in cell</td></tr>
      <tr><td>4</td><td>Ally tower HP</td><td>Fraction of max HP</td></tr>
      <tr><td>5</td><td>Enemy tower HP</td><td>Fraction of max HP</td></tr>
      <tr><td>6</td><td>Spell effects</td><td>Additive count</td></tr>
    </tbody>
  </table>

  <p>The 18x32 grid discretizes the arena into cells. Units are placed at the cell corresponding to their bounding box center. When multiple units map to the same cell, a collision resolution algorithm (using <code>scipy.spatial.distance.cdist</code>, inspired by KataCR's <code>PositionFinder</code>) displaces units to the nearest free cell. Enemy units are placed first, then ally units, ensuring enemy positions are preserved when conflicts arise.</p>

  <h4 class="sub-title">Vector Observation -- (23,) float32</h4>

  <table>
    <thead>
      <tr><th>Index</th><th>Feature</th><th>Normalization</th></tr>
    </thead>
    <tbody>
      <tr><td>0</td><td>Elixir</td><td>/ 10</td></tr>
      <tr><td>1</td><td>Time remaining</td><td>/ 300</td></tr>
      <tr><td>2</td><td>Is overtime</td><td>Binary</td></tr>
      <tr><td>3-5</td><td>Player tower HP (king, left princess, right princess)</td><td>Fraction of max</td></tr>
      <tr><td>6-8</td><td>Enemy tower HP</td><td>Fraction of max</td></tr>
      <tr><td>9-10</td><td>Player/enemy tower counts</td><td>/ 3</td></tr>
      <tr><td>11-14</td><td>Card present flags</td><td>Binary</td></tr>
      <tr><td>15-18</td><td>Card class indices</td><td>/ num_classes</td></tr>
      <tr><td>19-22</td><td>Card elixir costs</td><td>/ 10</td></tr>
    </tbody>
  </table>

</div>
</section>


<!-- Action Space -->

<section class="reveal">
<div class="sp">
  <h3>Action Space</h3>

  <p>We use a flat discrete action space:</p>

  <p>$$|\mathcal{A}| = 4 \times 18 \times 32 + 1 = 2305$$</p>

  <p>Each action index encodes which of the 4 hand cards to play, and which of the 576 arena grid cells (18 columns x 32 rows) to place it at, plus one <strong>no-op action</strong> (index 2304) for choosing to wait:</p>

  <h4 class="sub-title">Action Masking</h4>

  <p>Not all 2305 actions are valid at any given time. The <code>StateEncoder.action_mask()</code> function produces a boolean mask that disables:</p>

  <ul>
    <li>Actions for <strong>empty card slots</strong> (no card detected)</li>
    <li>Actions for cards that <strong>cost more elixir</strong> than currently available</li>
    <li>The no-op action is always valid</li>
  </ul>

  <p>This mask is passed to Stable Baselines 3's <code>MaskablePPO</code> to constrain the policy to only valid actions.</p>

  <h4 class="sub-title">Action Execution</h4>

  <p>The <code>ActionExecutor</code> decodes an action index back to (card_id, col, row), converts grid coordinates to normalized screen coordinates via <code>cell_to_norm()</code>, then executes two PyAutoGUI mouse clicks:</p>

  <ol>
    <li>Click on the <strong>card slot</strong> position</li>
    <li>Click on the <strong>arena placement</strong> position (with a 150ms delay between clicks to allow the game to register the card selection)</li>
  </ol>

</div>
</section>


<!-- Behavior Cloning Pipeline -->

<section class="reveal">
<div class="sp">
  <h3>Behavior Cloning Pipeline</h3>

  <p>Our learning approach follows a two-phase strategy: first train a policy via behavior cloning (BC) on human expert demonstrations, then fine-tune with PPO. The BC pipeline consists of four implemented modules.</p>

  <div class="step-card">
    <p><strong>1. Click Logger</strong> (<code>click_logger.py</code>)</p>
    <p>An OS-level mouse capture system using <code>pynput</code>. It implements a state machine that pairs card slot clicks with arena placement clicks:</p>
<pre style="margin: 0.5rem 0 0.25rem;">
idle -> mouse DOWN on card slot -> card_selected(card_id)
card_selected -> mouse RELEASE in arena -> emit action, return to idle
card_selected -> different card click -> update card_id (changed mind)
</pre>
    <p>This produces pre-paired actions (card_id, x_norm, y_norm) with timestamps, avoiding the need for post-hoc action inference from vision alone.</p>
  </div>

  <div class="step-card">
    <p><strong>2. Screen Capture</strong> (<code>screen_capture.py</code>)</p>
    <p>Threaded <code>mss</code> capture at configurable FPS (default 2 FPS). Saves JPEG screenshots with a <code>frames.jsonl</code> manifest containing timestamps for synchronization.</p>
  </div>

  <div class="step-card">
    <p><strong>3. Action Builder</strong> (<code>action_builder.py</code>)</p>
    <p>Converts raw click events to discrete action indices. The <code>build_action_timeline()</code> function assigns each expert action to its nearest frame by timestamp, labels all other frames as no-op (index 2304), producing a per-frame action sequence.</p>
  </div>

  <div class="step-card">
    <p><strong>4. Dataset Builder</strong> (<code>dataset_builder.py</code>)</p>
    <p>Processes recording sessions into training-ready <code>.npz</code> files. For each frame, it runs the perception pipeline (StateBuilder + StateEncoder) to produce observation tensors, pairs them with the action labels, and applies <strong>no-op downsampling</strong> (keeping 15% of no-op frames) to address the ~90% class imbalance between waiting and acting.</p>
  </div>

  <h4 class="sub-title">Output Format</h4>

  <table>
    <thead>
      <tr><th>Key</th><th>Shape</th><th>Description</th></tr>
    </thead>
    <tbody>
      <tr><td><code>obs_arena</code></td><td>(N, 32, 18, 7)</td><td>Arena grid observations</td></tr>
      <tr><td><code>obs_vector</code></td><td>(N, 23)</td><td>Vector observations</td></tr>
      <tr><td><code>actions</code></td><td>(N,)</td><td>Action indices (int64)</td></tr>
      <tr><td><code>masks</code></td><td>(N, 2305)</td><td>Valid action masks</td></tr>
      <tr><td><code>timestamps</code></td><td>(N,)</td><td>Frame timestamps</td></tr>
    </tbody>
  </table>

</div>
</section>


<!-- BC Model Architecture -->

<section class="reveal">
<div class="sp">
  <h3>BC Model Architecture (Trained)</h3>

  <p>The BC model uses a custom feature extractor feeding into a hierarchical 3-head policy. <strong>Total parameters: 267,950.</strong></p>

  <h4 class="sub-title">CRFeatureExtractor (192-dim output)</h4>

  <div class="two-col">
    <div class="step-card">
      <p><strong>Arena branch (128-dim)</strong></p>
      <p>The class ID channel is extracted, denormalized to integer, and passed through <code>nn.Embedding(155, 8)</code> to produce learned unit representations. These are concatenated with the remaining 5 continuous channels to form a (B, 13, 32, 18) tensor, processed by three Conv2d layers (13->32->64->128) with BatchNorm, ReLU, and progressive MaxPool, ending in AdaptiveAvgPool.</p>
    </div>
    <div class="step-card">
      <p><strong>Vector branch (64-dim)</strong></p>
      <p>Card class indices (4 slots) are each embedded via <code>nn.Embedding(9, 8)</code>, concatenated with 19 scalar features, then processed by two linear layers (51->64->64).</p>
    </div>
  </div>

  <p>The two branches produce a <strong>192-dim concatenated feature vector</strong>.</p>

  <h4 class="sub-title">BCPolicy (Hierarchical 3-Head Decomposition)</h4>

  <ul>
    <li><strong>Play head:</strong> Binary classifier (play a card vs. no-op) -- directly addresses the 70:30 noop imbalance</li>
    <li><strong>Card head:</strong> 4-way classifier (which card slot to play)</li>
    <li><strong>Position head:</strong> FiLM-conditioned 576-way classifier (which grid cell), with per-card spatial modulation</li>
  </ul>

  <p>The hierarchical decomposition was a critical design choice. A flat 2305-way softmax collapsed to always-noop during early experiments because ~3,000 noop examples vastly outweighed ~547 action examples spread across 2,304 placement classes (0.24 examples per class on average). The hierarchical approach splits this into solvable sub-problems.</p>

  <h4 class="sub-title">Training Data and Configuration</h4>

  <div class="two-col">
    <div>
      <table>
        <thead><tr><th>Training Data</th><th>Value</th></tr></thead>
        <tbody>
          <tr><td>Matches recorded</td><td>40</td></tr>
          <tr><td>Total frames (after downsampling)</td><td>5,366</td></tr>
          <tr><td>Action frames</td><td>~1,600 (30%)</td></tr>
          <tr><td>No-op frames</td><td>~3,766 (70%)</td></tr>
          <tr><td>No-op downsampling rate</td><td>15% retention</td></tr>
          <tr><td>Train/val split</td><td>80/20 file-level</td></tr>
          <tr><td>Data augmentation</td><td>Horizontal flip (2x)</td></tr>
        </tbody>
      </table>
    </div>
    <div>
      <table>
        <thead><tr><th>Hyperparameter</th><th>Value</th></tr></thead>
        <tbody>
          <tr><td>Optimizer</td><td>AdamW</td></tr>
          <tr><td>Learning rate</td><td>3e-4 (cosine to 1e-5)</td></tr>
          <tr><td>Weight decay</td><td>1e-4</td></tr>
          <tr><td>Batch size</td><td>64</td></tr>
          <tr><td>Epochs</td><td>31 (early stop on val F1)</td></tr>
          <tr><td>Play loss</td><td>Weighted BCE (0.3/3.0)</td></tr>
          <tr><td>Card loss</td><td>Cross-entropy</td></tr>
          <tr><td>Position loss</td><td>Label-smoothed CE</td></tr>
        </tbody>
      </table>
    </div>
  </div>

</div>
</section>


<!-- PPO Fine-Tuning -->

<section class="reveal">
<div class="sp">
  <h3>PPO Fine-Tuning (In Progress)</h3>

  <p>The PPO module is fully implemented and tested (41 tests passing). Training has begun with Phase 1 (frozen feature extractor) on live Clash Royale matches. Early runs focus on validating the environment wrapper, reward signal, and action execution loop. We are actively iterating on the training configuration and will report full results in the final report.</p>

  <h4 class="sub-title">ClashRoyaleEnv (Gymnasium-compatible)</h4>

  <p>Wraps live screen capture, YOLO detection, state encoding, and action execution into a standard Gymnasium environment:</p>

  <ul>
    <li><strong>2 Hz decision frequency</strong> (500ms step interval)</li>
    <li>Automatic phase detection (loading, battle, overtime, end)</li>
    <li>Action masking so only valid card placements are allowed</li>
  </ul>

  <h4 class="sub-title">Reward Function</h4>

  <table>
    <thead>
      <tr><th>Component</th><th>Value</th><th>Trigger</th></tr>
    </thead>
    <tbody>
      <tr><td>Enemy tower destroyed</td><td style="color: var(--cr-green-dark); font-weight: 600;">+10.0</td><td>Enemy tower count decreases</td></tr>
      <tr><td>Ally tower destroyed</td><td style="color: var(--cr-crimson); font-weight: 600;">-10.0</td><td>Ally tower count decreases</td></tr>
      <tr><td>Game win</td><td style="color: var(--cr-green-dark); font-weight: 600;">+30.0</td><td>Enemy king tower destroyed</td></tr>
      <tr><td>Game loss</td><td style="color: var(--cr-crimson); font-weight: 600;">-30.0</td><td>Ally king tower destroyed</td></tr>
      <tr><td>Draw</td><td style="color: var(--cr-crimson); font-weight: 600;">-5.0</td><td>Time expires, equal towers</td></tr>
      <tr><td>Survival bonus</td><td style="color: var(--cr-green-dark); font-weight: 600;">+0.02</td><td>Every step</td></tr>
      <tr><td>Elixir waste penalty</td><td style="color: var(--cr-crimson); font-weight: 600;">-0.05</td><td>Elixir >= 9.5</td></tr>
    </tbody>
  </table>

  <p>The reward function operates on observation deltas -- comparing tower counts between consecutive steps to detect tower destruction events.</p>

  <h4 class="sub-title">PPO Hyperparameters</h4>

  <div class="two-col">
    <div>
      <table>
        <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
        <tbody>
          <tr><td>Algorithm</td><td>MaskablePPO (sb3-contrib)</td></tr>
          <tr><td>Policy</td><td>MultiInputPolicy</td></tr>
          <tr><td>Features dim</td><td>192 (CRFeatureExtractor)</td></tr>
          <tr><td>Policy layers</td><td>[128, 64]</td></tr>
          <tr><td>Value layers</td><td>[128, 64]</td></tr>
          <tr><td>n_steps</td><td>512</td></tr>
          <tr><td>Batch size</td><td>64</td></tr>
        </tbody>
      </table>
    </div>
    <div>
      <table>
        <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
        <tbody>
          <tr><td>n_epochs</td><td>10</td></tr>
          <tr><td>Clip range</td><td>0.1</td></tr>
          <tr><td>Gamma</td><td>0.99</td></tr>
          <tr><td>GAE lambda</td><td>0.95</td></tr>
          <tr><td>Entropy coeff</td><td>0.01</td></tr>
          <tr><td>Max grad norm</td><td>0.5</td></tr>
          <tr><td></td><td></td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <h4 class="sub-title">Training Plan</h4>

  <p><strong>Phase 1</strong> (currently in progress) freezes the BC-pretrained feature extractor and trains only the PPO policy and value heads (lr=1e-4).</p>

  <p><strong>Phase 2</strong> will unfreeze the full network for end-to-end fine-tuning (lr=3e-5). We plan to continue scaling up the number of training games through the remainder of the quarter.</p>

</div>
</section>


<hr class="divider">


<!-- ============================================================ -->
<!-- EVALUATION                                                    -->
<!-- ============================================================ -->

<section class="reveal">
<h2 class="section-title">Evaluation</h2>
</section>


<!-- Object Detection Performance -->

<section class="reveal">
<div class="sp">
  <h3>Object Detection Performance</h3>

  <p><strong>Overall metrics</strong> (v12 model, YOLOv8s, 50 epochs, imgsz=960):</p>

  <div class="metric-row">
    <div class="metric-box gold"><span class="num">0.804</span><span class="label">mAP50</span></div>
    <div class="metric-box"><span class="num">0.567</span><span class="label">mAP50-95</span></div>
    <div class="metric-box"><span class="num">0.822</span><span class="label">Precision</span></div>
    <div class="metric-box"><span class="num">0.771</span><span class="label">Recall</span></div>
    <div class="metric-box"><span class="num">0.795</span><span class="label">F1 (conf=0.765)</span></div>
    <div class="metric-box"><span class="num">15.3</span><span class="label">FPS</span></div>
  </div>

  <h4 class="sub-title">Training Progression</h4>

  <p>mAP50 rose rapidly to 0.767 by epoch 5, reached 0.797 by epoch 10, then plateaued around 0.80 through epoch 50. Meanwhile, mAP50-95 continued improving slowly (0.477 at epoch 5 to 0.567 at epoch 50), indicating that bounding box precision kept improving even after detection recall saturated. No overfitting was observed -- validation loss never increased during training.</p>

  <table>
    <thead>
      <tr><th>Epoch</th><th>mAP50</th><th>mAP50-95</th></tr>
    </thead>
    <tbody>
      <tr><td>5</td><td>0.767</td><td>0.477</td></tr>
      <tr><td>10</td><td>0.797</td><td>0.520</td></tr>
      <tr><td>20</td><td>0.796</td><td>0.532</td></tr>
      <tr><td>30</td><td>0.804</td><td>0.550</td></tr>
      <tr><td>50</td><td><strong>0.804</strong></td><td><strong>0.567</strong></td></tr>
    </tbody>
  </table>


</div>
</section>


<!-- BC Training Results -->

<section class="reveal">
<div class="sp">
  <h3>BC Training Results</h3>

  <h4 class="sub-title">Best Validation Performance (epoch 6)</h4>

  <div class="metric-row">
    <div class="metric-box gold"><span class="num">0.322</span><span class="label">Action F1</span></div>
    <div class="metric-box"><span class="num">69.1%</span><span class="label">Action Recall</span></div>
    <div class="metric-box"><span class="num">21.0%</span><span class="label">Action Precision</span></div>
    <div class="metric-box"><span class="num">49.2%</span><span class="label">Overall Accuracy</span></div>
    <div class="metric-box"><span class="num">22.0%</span><span class="label">Card Accuracy</span></div>
  </div>

  <h4 class="sub-title">Training Progression</h4>

  <table>
    <thead>
      <tr><th>Metric</th><th>Epoch 1</th><th>Best</th><th>Epoch 31</th></tr>
    </thead>
    <tbody>
      <tr><td>Train loss (total)</td><td>7.90</td><td>5.61</td><td>5.61</td></tr>
      <tr><td>Train play loss</td><td>0.62</td><td>0.46</td><td>0.46</td></tr>
      <tr><td>Train card loss</td><td>1.39</td><td>0.74</td><td>0.74</td></tr>
      <tr><td>Train position loss</td><td>5.95</td><td>4.45</td><td>4.45</td></tr>
      <tr><td>Val loss</td><td>7.08</td><td>6.85 (ep. 4)</td><td>7.72</td></tr>
    </tbody>
  </table>


  <h4 class="sub-title">Analysis by Head</h4>

  <p>The <strong>play head</strong> reaches 69% recall, correctly identifying most card-play opportunities. The <strong>card head</strong> achieves 23% accuracy on 4-way classification -- limited by imbalanced card usage in the training data. The <strong>position head</strong> is the main bottleneck: with 576 grid cells and only ~1,600 action frames, predictions collapse to a narrow spatial band rather than learning precise placement. Overfitting appears after epoch 6, with validation loss diverging from training loss.</p>

  <div style="text-align: center; margin-top: 1rem;">
    <img src="{{ 'images/status/bc-evaluation.png' | relative_url }}" alt="BC model evaluation showing play/noop confusion matrix, card confusion matrix, placement heatmaps, and validation summary" style="max-width: 100%; border-radius: var(--border-radius); border: 2px solid var(--cr-brown-border);">
    <p style="font-size: 0.85rem; color: var(--cr-text-muted); margin-top: 0.5rem;">Validation results: the play/noop confusion matrix (left) shows the model catches most action frames but over-predicts play. The placement heatmaps (bottom) reveal spatial collapse -- the model predicts a tight cluster instead of the varied ground-truth positions.</p>
  </div>

</div>
</section>




<!-- Pipeline Test Coverage -->

<section class="reveal">
<div class="sp">
  <h3>Pipeline Test Coverage</h3>

  <p>All modules have been tested with <strong>153 unit tests</strong> across the full pipeline:</p>

  <table>
    <thead>
      <tr><th>Module</th><th>Tests</th><th>Status</th><th>Key Coverage</th></tr>
    </thead>
    <tbody>
      <tr><td>State Encoder</td><td>42</td><td><span class="elixir-badge">All passing</span></td><td>Grid encoding, collision resolution, action masking, edge cases</td></tr>
      <tr><td>Action Builder</td><td>46</td><td><span class="elixir-badge">All passing</span></td><td>Click pairing state machine, action encoding/decoding, timeline building</td></tr>
      <tr><td>Dataset Builder</td><td>24</td><td><span class="elixir-badge">All passing</span></td><td>Session processing, no-op downsampling, multi-session merging</td></tr>
      <tr><td>PPO Module</td><td>41</td><td><span class="elixir-badge">All passing</span></td><td>Environment wrapper, reward function, BC weight loading, action masking</td></tr>
      <tr><td><strong>Total</strong></td><td><strong>153</strong></td><td><span class="elixir-badge elixir-badge--gold">All passing</span></td><td></td></tr>
    </tbody>
  </table>

  <p>These tests validate the entire data flow from raw clicks through BC training and into the PPO environment, using mock GameState objects to isolate the encoding logic from the perception pipeline.</p>

</div>
</section>


<hr class="divider">


<!-- ============================================================ -->
<!-- REMAINING GOALS AND CHALLENGES                                -->
<!-- ============================================================ -->

<section class="reveal">
<h2 class="section-title">Remaining Goals and Challenges</h2>
</section>

<section class="reveal">
<div class="sp">
  <h3>Immediate Next Steps</h3>

  <div class="step-card">
    <p><strong>1. Continue PPO training.</strong> Phase 1 (frozen feature extractor) is underway and we are actively iterating on the training loop. Next is Phase 2 (full fine-tuning) and scaling up the number of live training games through the remainder of the quarter.</p>
  </div>

  <div class="step-card">
    <p><strong>2. Expand expert demonstration data.</strong> Recording additional expert matches will improve position head learning and provide more diverse card usage examples, strengthening both the BC baseline and PPO initialization.</p>
  </div>

</div>
</section>

<section class="reveal">
<div class="sp">
  <h3>Known Challenges</h3>

  <div class="challenge high">
    <h4>Position learning with limited data <span class="elixir-badge elixir-badge--crimson">High Impact</span></h4>
    <p>The BC position head has 576 possible grid cells but only ~1,600 action examples, yielding fewer than 3 examples per cell on average. The model learns spatial priors (e.g., "play near the bridge") but not precise tactical placement. More expert data (20-40 additional matches) and PPO fine-tuning should improve this -- the RL reward signal can teach spatial awareness that pure imitation cannot.</p>
  </div>

  <div class="challenge high">
    <h4>Object detection accuracy <span class="elixir-badge elixir-badge--crimson">High Impact</span></h4>
    <p>The current YOLOv8s model (mAP50=0.804) is trained on synthetic data, producing a domain gap with real gameplay. Missed detections and misclassifications propagate through the entire pipeline, degrading the agent's state observations and decision quality. Improving detection accuracy through fine-tuning on real gameplay frames and expanding the custom sprite pool is a priority.</p>
  </div>

  <div class="challenge med">
    <h4>No temporal context <span class="elixir-badge elixir-badge--gold">Medium Impact</span></h4>
    <p>Our state representation captures a single frame -- there is no memory of previous observations. This means the agent cannot track troop movement trajectories, estimate opponent elixir from deployment timing, or learn timing-dependent strategies. Adding frame stacking or an LSTM layer is a potential improvement for the final report.</p>
  </div>

  <div class="challenge low">
    <h4>Live inference latency <span class="elixir-badge">Low Impact</span></h4>
    <p>The full pipeline runs at ~50ms per frame during live inference, well within the 500ms budget at 2 Hz decision frequency. The bottleneck is action execution delay (~150ms for the two-click card placement sequence via PyAutoGUI), not perception or model inference.</p>
  </div>

</div>
</section>


<hr class="divider">


<!-- ============================================================ -->
<!-- RESOURCES USED                                                -->
<!-- ============================================================ -->

<section class="reveal">
<h2 class="section-title">Resources Used</h2>
</section>

<section class="reveal">
<div class="sp">
  <h3>Core Libraries and Frameworks</h3>

  <ul>
    <li><strong><a href="https://github.com/ultralytics/ultralytics">Ultralytics YOLOv8</a></strong> -- Object detection model, training infrastructure, and pretrained weights. We use the v8s architecture with custom extensions for belonging prediction.</li>
    <li><strong><a href="https://github.com/PaddlePaddle/PaddleOCR">PaddleOCR</a></strong> -- Optical character recognition for reading elixir count, match timer, and tower HP values from screen captures.</li>
    <li><strong><a href="https://github.com/DLR-RM/stable-baselines3">Stable Baselines 3</a></strong> -- RL library providing Gymnasium-compatible environment interfaces. Used for observation/action space definitions and planned PPO training.</li>
    <li><strong><a href="https://github.com/Stable-Baselines-Contrib/stable-baselines3-contrib">SB3-Contrib</a></strong> -- MaskablePPO implementation for action-masked reinforcement learning.</li>
    <li><strong><a href="https://pytorch.org/">PyTorch</a></strong> -- Deep learning framework for the card classifier, BC model, and custom YOLOv8 loss functions.</li>
    <li><strong><a href="https://opencv.org/">OpenCV</a></strong> -- Image processing, video frame extraction, and visualization.</li>
    <li><strong><a href="https://github.com/BoboTiG/python-mss">mss</a></strong> -- Cross-platform screen capture library for real-time frame acquisition.</li>
    <li><strong><a href="https://github.com/asweigart/pyautogui">PyAutoGUI</a></strong> -- Mouse automation for executing card placements in the game client.</li>
    <li><strong><a href="https://github.com/moses-palmer/pynput">pynput</a></strong> -- OS-level keyboard and mouse event monitoring for the click logger.</li>
    <li><strong><a href="https://flask.palletsprojects.com/">Flask</a></strong> -- Web framework powering the Quick-Seg sprite cutout tool.</li>
    <li><strong><a href="https://numpy.org/">NumPy</a></strong> and <strong><a href="https://scipy.org/">SciPy</a></strong> -- Numerical computation, array operations, and spatial distance calculations for collision resolution.</li>
  </ul>

</div>
</section>

<section class="reveal">
<div class="sp">
  <h3>Datasets and External Code</h3>

  <ul>
    <li><strong><a href="https://github.com/wty-yy/KataCR">KataCR</a></strong> (MIT License) -- We used KataCR's published dataset and adopted their 155-class label schema. Their idea of training on synthetic data inspired our own synthetic generation pipeline, which we customized for our needs (custom arena backgrounds, belonging-aware sprites, deck-specific augmentation).</li>
    <li><strong><a href="https://github.com/wty-yy/Clash-Royale-Detection-Dataset">Clash Royale Detection Dataset</a></strong> (MIT License) -- KataCR's dataset of 6,939 human-annotated real gameplay images. We use 1,388 as our validation set and maintain a forked version with 233 additional ally sprite cutouts for belonging-aware training.</li>
    <li><strong><a href="https://github.com/Pbatch/ClashRoyaleBuildABot">ClashRoyaleBuildABot</a></strong> -- Reference project for another Clash Royale AI agent. We consulted their approach to screen capture and card detection for design inspiration.</li>  </ul>

</div>
</section>

<section class="reveal">
<div class="sp">
  <h3>Platform</h3>

  <p><strong>Google Play Games (PC)</strong> -- Chosen over Android emulators (BlueStacks, LDPlayer) for native Windows performance and consistent rendering.</p>

  <h3 style="margin-top: 1.5rem;">AI Tool Usage</h3>

  <p>We used <strong>Claude</strong> (Anthropic) as an auxiliary tool at specific points in the project:</p>

  <ul>
    <li><strong>Boilerplate and scaffolding:</strong> Claude helped generate initial code templates for repetitive tasks such as evaluation scripts, data loading utilities, and config file setup. All generated code was reviewed and adapted by team members.</li>
    <li><strong>Documentation drafting:</strong> Claude assisted with drafting portions of design documents and this status report. All content was fact-checked against the codebase and rewritten where needed.</li>
    <li><strong>Debugging assistance:</strong> When troubleshooting integration issues (e.g., coordinate system mismatches between modules), we occasionally consulted Claude for diagnostic suggestions.</li>
  </ul>

  <p>Core technical work -- model architecture selection, training strategy, synthetic data pipeline design, action space formulation, and the behavior cloning pipeline -- was carried out by the team through iterative experimentation and analysis of Clash Royale gameplay.</p>

</div>
</section>
