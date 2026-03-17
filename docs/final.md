---
layout: default
title: Final Report
page_title: Final Report
page_subtitle: "The Elixir Optimizers -- Final Project Report"
wide: true
---

<style>
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
  .two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-lg);
    margin: var(--space-md) 0;
  }
  @media (max-width: 767px) {
    .two-col { grid-template-columns: 1fr; }
  }
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
  .sp code {
    font-size: 0.85rem;
    padding: 0.1em 0.35em;
  }
  .section-title {
    color: #fff;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    font-family: var(--font-heading);
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    margin-bottom: var(--space-md);
  }
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
  .sp table { margin: 0.75rem 0 1rem; font-size: 0.9rem; }
  .sp th { font-size: 0.75rem; padding: 0.4rem 0.75rem; }
  .sp td { padding: 0.4rem 0.75rem; }
  .fig { text-align: center; margin: 1rem 0; }
  .fig img { max-width: 100%; border-radius: var(--border-radius); }
  .fig .caption {
    font-size: 0.8rem;
    color: var(--cr-text-muted);
    margin-top: 0.4rem;
    font-style: italic;
  }
</style>

<!-- Video -->
<section class="reveal">
  <div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;border:2px solid var(--cr-brown-border);border-radius:var(--border-radius-lg);box-shadow:0 2px 8px rgba(0,0,0,0.08);margin-bottom:var(--space-lg);">
    <iframe src="https://www.youtube.com/embed/CjHgsb8ZUgk" title="The Elixir Optimizers - Final Report Video" style="position:absolute;top:0;left:0;width:100%;height:100%;border:0;" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
  </div>
</section>

<hr class="divider">


<!-- ============================================================ -->
<!-- PROJECT SUMMARY                                               -->
<!-- ============================================================ -->

<section class="reveal">
<h2 class="section-title" style="background: linear-gradient(135deg, var(--cr-sky-mid), var(--cr-blue-dark)); padding: var(--space-md) var(--space-lg); border-radius: var(--border-radius-lg);">Project Summary</h2>
<div class="sp">

  <p>We built an <strong>end-to-end reinforcement learning agent</strong> that plays Clash Royale -- a real-time strategy mobile game with 100M+ downloads -- using only raw screen pixels as input. Unlike every prior game-playing AI breakthrough (DQN on Atari, AlphaStar on StarCraft II, OpenAI Five on Dota 2), our agent has no access to a game API, simulator, or internal state. It must perceive the game through computer vision, reason about strategy, and act through mouse automation -- the hardest setting for game AI.</p>

  <h4 class="sub-title">Why It Matters / Why It's Hard</h4>

  <ul>
    <li><strong>No simulator or API</strong> -- all game state must be derived from pixels via real-time computer vision</li>
    <li><strong>Real-time decisions under partial observability</strong> -- fog of war hides parts of the arena, and enemy elixir count is never visible</li>
    <li><strong>Large structured action space</strong> -- 4 cards x 576 grid cells + no-op = 2,305 possible actions per frame</li>
    <li><strong>High visual complexity</strong> -- 155 entity classes with overlapping, animated units on a cluttered battlefield</li>
    <li><strong>Domain gap</strong> -- the detector is trained on synthetic sprite composites but must perform on real gameplay rendering</li>
    <li><strong>Sparse delayed rewards</strong> -- the match outcome (win/loss/draw) arrives only after roughly 3 minutes of continuous play</li>
  </ul>

  <h4 class="sub-title">Comparison with Prior Systems</h4>

  <p>Our work sits in the hardest quadrant of the game AI design space: pixel-only perception with no access to a simulator or game API.</p>

  <table>
    <thead>
      <tr><th>System</th><th>Game</th><th>Pixel Input?</th><th>Simulator?</th><th>API?</th></tr>
    </thead>
    <tbody>
      <tr><td>DQN (2015)</td><td>Atari</td><td>Yes</td><td>Yes</td><td>Yes</td></tr>
      <tr><td>AlphaStar (2019)</td><td>StarCraft II</td><td>No</td><td>Yes</td><td>Yes</td></tr>
      <tr><td>OpenAI Five (2019)</td><td>Dota 2</td><td>No</td><td>Yes</td><td>Yes</td></tr>
      <tr><td><strong>Ours (2026)</strong></td><td><strong>Clash Royale</strong></td><td><strong>Yes</strong></td><td><strong>No</strong></td><td><strong>No</strong></td></tr>
    </tbody>
  </table>

  <h4 class="sub-title">Our Approach -- Three Learned Systems</h4>

  <p>The agent is built from three learned systems chained together. A <strong>dual YOLOv8m detector</strong> (155 classes) converts raw pixels into structured detections. A <strong>state encoder</strong> transforms those detections, OCR readings, and card classifications into a fixed-size game state tensor. A <strong>behavior cloning policy</strong> trained on expert demonstrations provides an initial strategy, which is then refined through <strong>PPO fine-tuning</strong> on live matches -- producing an agent that can play and win real games.</p>

  <h4 class="sub-title">Project Goals</h4>

  <table>
    <thead>
      <tr><th>Tier</th><th>Goal</th><th>Status</th></tr>
    </thead>
    <tbody>
      <tr><td>Minimum</td><td>End-to-end pipeline functional: pixels in, actions out</td><td>Achieved</td></tr>
      <tr><td>Target</td><td>Beat in-game Trainer AI with >50% win rate</td><td>Exceeded -- 30% vs real humans</td></tr>
      <tr><td>Moonshot</td><td>Competitive on ranked ladder</td><td>Not attempted</td></tr>
    </tbody>
  </table>

  <div class="step-card">
    <p><strong>Note:</strong> Reaching a 30% win rate against real human opponents is harder and more meaningful than beating the scripted Trainer AI. We exceeded our target goal in a harder setting.</p>
  </div>

  <div class="metric-row">
    <div class="metric-box gold"><span class="num">0.804</span><span class="label">mAP50</span></div>
    <div class="metric-box"><span class="num">155</span><span class="label">Detection Classes</span></div>
    <div class="metric-box gold"><span class="num">30%</span><span class="label">Win Rate</span></div>
    <div class="metric-box"><span class="num">237ms</span><span class="label">Latency</span></div>
    <div class="metric-box"><span class="num">106</span><span class="label">Live Episodes</span></div>
    <div class="metric-box"><span class="num">153</span><span class="label">Unit Tests</span></div>
  </div>

</div>
</section>

<hr class="divider">

<!-- ============================================================ -->
<!-- APPROACH                                                      -->
<!-- ============================================================ -->

<section class="reveal">
  <h2 class="section-title" style="background: linear-gradient(135deg, var(--cr-sky-mid), var(--cr-blue-dark)); padding: var(--space-md) var(--space-lg); border-radius: var(--border-radius-lg);">Approach</h2>
</section>


<!-- 3a. System Architecture -->
<section class="reveal">
<div class="sp">
  <h3>3a. System Architecture</h3>

  <p>Our system is an <strong>external observer</strong> - there is no game memory injection, custom simulator, or internal API. It operates in two modes: <strong>data collection</strong> (recording expert demonstrations for behavior cloning) and <strong>live inference</strong> (autonomous play). In both modes the same perception pipeline converts raw pixels into structured game state; the difference is whether a human or a policy network decides what to do next.</p>

  <div class="fig">
    <img src="{{ "images/pipeline-architecture.svg" | relative_url }}" alt="Full pipeline architecture diagram showing perception, state encoding, and decision modules">
    <p class="caption">Figure 1. End-to-end pipeline architecture - from screen capture through detection, state encoding, and action execution.</p>
  </div>

  <h4 class="sub-title">Pipeline Stages</h4>

  <div class="step-card">
    <p><strong>1. Screen Capture (mss)</strong></p>
    <p>Captures the Google Play Games window at 2-4 FPS. Platform-native rendering on Windows ensures consistent frame geometry across sessions.</p>
  </div>

  <div class="step-card">
    <p><strong>2. Dual YOLOv8m Detection</strong></p>
    <p>155 classes split across two specialized models. D1 handles small and medium units; D2 handles large units and structures. Both run in parallel and their outputs are merged.</p>
  </div>

  <div class="step-card">
    <p><strong>3. PaddleOCR + Card Classifier</strong></p>
    <p>PaddleOCR reads elixir count, match timer, and tower HP from calibrated screen regions. A MiniResNet classifier identifies the four cards currently in hand.</p>
  </div>

  <div class="step-card">
    <p><strong>4. StateBuilder</strong></p>
    <p>Assembles a structured game state tensor from all perception outputs: a 32x18x6 arena grid encoding unit positions and HP, plus a 23-dimensional vector of global features.</p>
  </div>

  <div class="step-card">
    <p><strong>5. Policy Network</strong></p>
    <p>A convolutional + MLP network warm-started via behavior cloning on expert demonstrations, then fine-tuned with PPO on live match outcomes.</p>
  </div>

  <div class="step-card">
    <p><strong>6. Action Executor</strong></p>
    <p>PyAutoGUI mouse automation translates the policy's discrete action into a card selection and grid placement click, with a 150ms inter-action delay to respect game animation timing.</p>
  </div>

</div>
</section>


<!-- 3b. Object Detection with YOLOv8 -->
<section class="reveal">
<div class="sp">
  <h3>3b. Object Detection with YOLOv8</h3>

  <h4 class="sub-title">Why Dual YOLOv8m?</h4>

  <p>Our first approach used a single YOLOv8s model (11.2M parameters) for all 155 classes plus belonging prediction. This failed in several ways:</p>

  <ul>
    <li><strong>Capacity bottleneck</strong> - 155+ output channels per anchor overwhelmed the small detection head</li>
    <li><strong>Scale conflict</strong> - small sprites (skeletons, bats) need fine-grained features while large sprites (golems, towers) need broad receptive fields</li>
    <li><strong>Confusion clusters</strong> - visually similar classes (wizard/witch, barbarian variants) competed for limited capacity</li>
    <li><strong>Belonging interference</strong> - the ally/enemy classification task competed with object detection for the same parameters</li>
  </ul>

  <p>Our solution: <strong>split into two specialized YOLOv8m models</strong> (~25.9M parameters each, ~51.8M total). D1 handles approximately 85 classes focused on small and medium units. D2 handles approximately 85 classes focused on large units and structures. 13 base classes appear in both models for cross-reference during NMS merging.</p>

  <table>
    <thead>
      <tr><th></th><th>Single YOLOv8s</th><th>Dual YOLOv8m</th></tr>
    </thead>
    <tbody>
      <tr><td>Parameters</td><td>11.2M</td><td>~51.8M (2x 25.9M)</td></tr>
      <tr><td>Classes per head</td><td>156 (all + belonging)</td><td>~82-85 each (13 shared base)</td></tr>
      <tr><td>Specialization</td><td>None</td><td>D1: small/medium, D2: large/structures</td></tr>
      <tr><td>mAP50</td><td>0.756</td><td><strong>0.804</strong> (+6%)</td></tr>
      <tr><td>Hallucinations</td><td>12</td><td><strong>0</strong></td></tr>
      <tr><td>Detections/frame</td><td>24.1</td><td><strong>30.2</strong> (+25%)</td></tr>
      <tr><td>Unique classes seen</td><td>68</td><td><strong>85</strong> (+25%)</td></tr>
    </tbody>
  </table>

  <h4 class="sub-title">155-Class Taxonomy</h4>

  <table>
    <thead>
      <tr><th>Category</th><th>Examples</th><th>Notes</th></tr>
    </thead>
    <tbody>
      <tr><td>Towers</td><td>King Tower, Princess Tower (all levels)</td><td>HP bars, damage states</td></tr>
      <tr><td>Ground units</td><td>Knight, Prince, Royal Giant, ...</td><td>Ally and enemy variants</td></tr>
      <tr><td>Flying units</td><td>Princess, Flying Machine, ...</td><td>Overlapping with ground layer</td></tr>
      <tr><td>Spells</td><td>Fireball, Zap, Arrows, ...</td><td>Transient visual effects</td></tr>
      <tr><td>UI elements</td><td>Elixir bar, timer, card slots</td><td>Fixed screen positions</td></tr>
    </tbody>
  </table>

  <h4 class="sub-title">Synthetic Data Pipeline</h4>

  <p>No labeled real gameplay data exists for all 155 classes. Following the KataCR approach, we built a fully synthetic data generation pipeline that composes sprite cutouts onto arena backgrounds.</p>

  <div class="step-card">
    <p><strong>1. Arena Background</strong></p>
    <p>Randomized arena variants provide visual diversity in lighting, color, and terrain.</p>
  </div>

  <div class="step-card">
    <p><strong>2. Tower Sprites</strong></p>
    <p>Placed at fixed positions matching real game layout, with level and damage state variations.</p>
  </div>

  <div class="step-card">
    <p><strong>3. Unit Sprite Placement</strong></p>
    <p>Up to 40 unit sprites per image, placed on an 18x32 occupancy grid to prevent overlap.</p>
  </div>

  <div class="step-card">
    <p><strong>4. Noise Sprites (Hard Negatives)</strong></p>
    <p>25% of placed sprites are noise - irrelevant visual elements that teach the detector what <em>not</em> to detect. This single technique eliminated hallucinations from 50 per evaluation to 0.</p>
  </div>

  <div class="step-card">
    <p><strong>5. Class Balancing</strong></p>
    <p>Inverse frequency weighting ensures rare classes appear often enough for the detector to learn them.</p>
  </div>

  <div class="step-card">
    <p><strong>6. Augmentation</strong></p>
    <p>Color filters, scale/stretch transforms, and HSV shifts bridge the domain gap between synthetic composites and real rendered frames.</p>
  </div>

  <p>Each training epoch generates <strong>20,000 unique synthetic images</strong> - the model never sees the same composition twice.</p>

  <h4 class="sub-title">Training Configuration</h4>

  <table>
    <thead>
      <tr><th>Parameter</th><th>Value</th></tr>
    </thead>
    <tbody>
      <tr><td>Base model</td><td>YOLOv8m pretrained on COCO</td></tr>
      <tr><td>Image size</td><td>960x960</td></tr>
      <tr><td>Batch size</td><td>16</td></tr>
      <tr><td>Epochs</td><td>50 (early stopping patience=15)</td></tr>
      <tr><td>Optimizer</td><td>SGD, lr=0.01, linear decay to 0.01</td></tr>
      <tr><td>Augmentation</td><td>degrees=5, scale=0.5, fliplr=0.5, erasing=0.4, HSV shifts</td></tr>
      <tr><td>Mosaic / Mixup</td><td>Disabled</td></tr>
      <tr><td>AMP</td><td>Enabled</td></tr>
      <tr><td>Validation</td><td>1,388 human-annotated real images from KataCR</td></tr>
    </tbody>
  </table>

  <h4 class="sub-title">Belonging Prediction (Ally vs Enemy)</h4>

  <p>Knowing <em>which side</em> a unit belongs to is as important as knowing <em>what</em> it is. We extended YOLOv8 with a custom belonging head in our <code>CRDetectionModel</code> class.</p>

  <ul>
    <li><strong>Architecture</strong> - the prediction tensor has C+1 channels, where the last channel is a binary belonging score (0 = ally, 1 = enemy)</li>
    <li><strong>Loss</strong> - BCE loss folded into the classification loss via YOLOv8's Task-Aligned Assigner</li>
    <li><strong>NMS output</strong> - custom NMS produces 7-column detections: <code>(x1, y1, x2, y2, conf, cls, belong)</code></li>
    <li><strong>Training data</strong> - 383 custom ally sprite cutouts created with our Quick-Seg tool, paired against the existing enemy sprites</li>
  </ul>

  <h4 class="sub-title">Quick-Seg: Browser-Based Sprite Cutout Tool</h4>

  <div class="two-col">
    <div>
      <p>To create the 233 ally sprite cutouts needed for belonging training, we built <strong>Quick-Seg</strong> - a browser-based annotation tool.</p>
      <ul>
        <li>Flask backend with OpenCV and NumPy processing</li>
        <li>Frame-by-frame video navigation</li>
        <li>Freehand lasso selection with add/subtract modes</li>
        <li>Exports transparent RGBA PNGs organized by class folder</li>
        <li>Open-sourced at <a href="https://github.com/weihaog1/Quick-Seg">github.com/weihaog1/Quick-Seg</a></li>
      </ul>
    </div>
    <div>
      <div class="fig">
        <img src="{{ "images/status/quick-seg-screeshot.png" | relative_url }}" alt="Quick-Seg browser interface showing lasso selection on a game frame">
        <p class="caption">Figure 2. Quick-Seg interface - freehand lasso selection for extracting transparent sprite cutouts from gameplay footage.</p>
      </div>
    </div>
  </div>

  <h4 class="sub-title">Detection in Action</h4>

  <div class="fig">
    <img src="{{ "images/status/detection_demo.gif" | relative_url }}" alt="Animated demo of real-time detection with bounding boxes and belonging labels on live gameplay">
    <p class="caption">Figure 3. Real-time detection on live gameplay - bounding boxes with class labels and ally/enemy belonging indicators.</p>
  </div>

</div>
</section>


<!-- 3c. Card Classifier + OCR -->
<section class="reveal">
<div class="sp">
  <h3>3c. Card Classifier + OCR</h3>

  <div class="two-col">
    <div>
      <h4 class="sub-title">Card Classifier</h4>

      <p>Identifies the four cards currently in the player's hand. Because deck composition is known ahead of time, this is an 8-class closed-set problem.</p>

      <ul>
        <li><strong>Architecture</strong> - MiniResNet (~156K parameters), a 4-block residual network with 8-class softmax output</li>
        <li><strong>Training data</strong> - 8 reference images (one per card in the deck), expanded to 4,000 augmented samples per epoch (500 per class)</li>
        <li><strong>Augmentation</strong> - RandomAffine, ColorJitter, GaussianBlur, RandomErasing, and greyout transforms to handle the dimmed appearance of unplayable cards</li>
        <li><strong>Optimizer</strong> - AdamW with CosineAnnealing learning rate schedule</li>
        <li><strong>Result</strong> - high classification accuracy on live gameplay frames; confidence threshold of 0.6 used to flag empty/unrecognized slots</li>
      </ul>
    </div>
    <div>
      <h4 class="sub-title">OCR (PaddleOCR)</h4>

      <p>Reads numeric game state values from fixed screen regions, calibrated for a 540x960 base resolution.</p>

      <ul>
        <li><strong>Match timer</strong> - ~85% accuracy</li>
        <li><strong>Elixir count</strong> - ~90% accuracy</li>
        <li><strong>Tower HP</strong> - reads all visible tower health bars</li>
        <li><strong>8 screen regions</strong> calibrated for consistent capture geometry (timer, elixir, 6 tower HP bars)</li>
      </ul>

      <p><strong>Preprocessing:</strong> CLAHE contrast enhancement followed by binary thresholding and inversion to maximize OCR reliability.</p>

      <p><strong>Error correction:</strong> regex-based post-processing fixes common OCR mistakes (O to 0, I to 1) with range validation to reject impossible values.</p>

      <p>OCR adds approximately <strong>50ms</strong> to the pipeline latency.</p>
    </div>
  </div>

</div>
</section>


<!-- 3d. State Representation -->
<section class="reveal">
<div class="sp">
  <h3>3d. State Representation</h3>

  <p>The StateBuilder converts raw perception outputs into a fixed-size tensor that the policy network can consume. The state has two components:</p>

  <div class="two-col">
    <div>
      <h4 class="sub-title">Arena Grid (32 x 18 x 7 channels)</h4>

      <p>A spatial representation of the battlefield. Each cell in the 32x18 grid corresponds to a region of the arena. Seven channels encode unit presence, tower health, and spell effects:</p>

      <table>
        <thead>
          <tr><th>Channel</th><th>Content</th></tr>
        </thead>
        <tbody>
          <tr><td>0</td><td>Ally ground units</td></tr>
          <tr><td>1</td><td>Ally flying units</td></tr>
          <tr><td>2</td><td>Enemy ground units</td></tr>
          <tr><td>3</td><td>Enemy flying units</td></tr>
          <tr><td>4</td><td>Ally tower HP (fraction)</td></tr>
          <tr><td>5</td><td>Enemy tower HP (fraction)</td></tr>
          <tr><td>6</td><td>Spell effects (additive count)</td></tr>
        </tbody>
      </table>

      <p>Each cell holds the normalized HP of the strongest unit present in that position. This gives the policy a spatial understanding of force distribution across the arena.</p>
    </div>
    <div>
      <h4 class="sub-title">Vector Features (23 dimensions)</h4>

      <p>Global game state values that do not have a spatial position on the arena:</p>

      <table>
        <thead>
          <tr><th>Indices</th><th>Feature</th><th>Normalization</th></tr>
        </thead>
        <tbody>
          <tr><td>0</td><td>Elixir</td><td>/10</td></tr>
          <tr><td>1</td><td>Time remaining</td><td>/300</td></tr>
          <tr><td>2</td><td>Is overtime</td><td>Binary</td></tr>
          <tr><td>3-5</td><td>Player tower HP</td><td>Fraction</td></tr>
          <tr><td>6-8</td><td>Enemy tower HP</td><td>Fraction</td></tr>
          <tr><td>9-10</td><td>Tower counts</td><td>/3</td></tr>
          <tr><td>11-14</td><td>Card present flags</td><td>Binary</td></tr>
          <tr><td>15-18</td><td>Card class indices</td><td>/num_classes</td></tr>
          <tr><td>19-22</td><td>Card elixir costs</td><td>/10</td></tr>
        </tbody>
      </table>

      <p>All values are normalized to the <strong>[0, 1]</strong> range to ensure stable gradient flow during training.</p>
    </div>
  </div>

</div>
</section>


<!-- 3e. Action Space -->
<section class="reveal">
<div class="sp">
  <h3>3e. Action Space</h3>

  <p>The agent uses a flat discrete action space of <strong>Discrete(2305)</strong> actions: 4 card slots multiplied by 576 grid cells (32 rows x 18 columns), plus a single no-op action.</p>

  <h4 class="sub-title">Action Encoding</h4>

  <pre><code>action = card_slot * 576 + row * 18 + col
no-op  = 2304</code></pre>

  <p>Given an action index, the executor decodes the card slot and grid coordinates, selects the card from the hand UI, then clicks the corresponding position on the arena.</p>

  <h4 class="sub-title">Dynamic Action Masking</h4>

  <p>Not all 2,305 actions are valid at any given moment. The policy applies dynamic masking before sampling:</p>

  <ul>
    <li><strong>Empty card slots</strong> - if a card slot is empty (waiting for cycle), all 576 positions for that slot are masked</li>
    <li><strong>Insufficient elixir</strong> - if the player lacks elixir to play a card, all positions for that card are masked</li>
    <li><strong>No-op always valid</strong> - action 2304 is never masked; the agent can always choose to wait</li>
  </ul>

  <pre><code>logits[invalid_mask] = float('-inf')
probs = softmax(logits)   # invalid actions get probability 0</code></pre>

  <p>This masking eliminates wasted exploration on unplayable actions, focusing the policy's learning capacity on decisions that can actually affect the game.</p>

</div>
</section>

<hr class="divider">
<!-- ============================================================ -->
<!-- APPROACH 3f: BEHAVIOR CLONING                                -->
<!-- ============================================================ -->

<section class="reveal">
<h3 class="section-title">3f. Behavior Cloning</h3>
<div class="sp">

  <h4 class="sub-title">Data Collection Pipeline</h4>

  <p>Expert demonstrations were collected from live gameplay using an OS-level click logger built on <code>pynput</code> and <code>mss</code>. The logger runs a state machine that pairs card-slot clicks with arena placements: <strong>idle -> card_selected(k)</strong> on mouse-down in a card slot, then <strong>emit(k, x_norm, y_norm)</strong> on mouse-release inside the arena, returning to idle. Screen capture runs in a separate thread at 2 FPS. Four modules handle the full pipeline from raw input to training-ready data:</p>

  <div class="two-col">
    <div class="step-card">
      <p><strong>1. ClickLogger</strong></p>
      <p>A <code>pynput</code> state machine that pairs card-slot mouse-down events with arena mouse-release events. Outputs <code>(card_id, x_norm, y_norm)</code> tuples with millisecond timestamps.</p>
    </div>
    <div class="step-card">
      <p><strong>2. ScreenCapture</strong></p>
      <p>Threaded <code>mss</code> daemon capturing at 2 FPS with JPEG compression (quality 85). Writes a <code>frames.jsonl</code> manifest mapping each frame to its capture timestamp.</p>
    </div>
    <div class="step-card">
      <p><strong>3. ActionBuilder</strong></p>
      <p>Converts raw click events to <code>Discrete(2305)</code> indices. Assigns each action to the nearest frame by timestamp. Labels all remaining frames as no-op (index 2304).</p>
    </div>
    <div class="step-card">
      <p><strong>4. DatasetBuilder</strong></p>
      <p>Runs the full perception pipeline per frame (YOLOv8 + OCR + card classifier). Applies 15% no-op retention to reduce class imbalance. Saves output as <code>.npz</code> files.</p>
    </div>
  </div>

  <p>We played 100 expert matches total, recording the last 40 after the logger was stable and our play had improved:</p>

  <table>
    <thead>
      <tr><th>Metric</th><th>Value</th></tr>
    </thead>
    <tbody>
      <tr><td>Expert matches played</td><td>100</td></tr>
      <tr><td>Matches recorded</td><td>40 (last 40)</td></tr>
      <tr><td>Total frames</td><td>5,366</td></tr>
      <tr><td>Action frames</td><td>~1,600 (30%)</td></tr>
      <tr><td>No-op frames</td><td>~3,766 (70%)</td></tr>
      <tr><td>After no-op downsample (15%)</td><td>~2,165 effective frames</td></tr>
      <tr><td>Output format</td><td>5 .npz files, 80/20 train/val split</td></tr>
    </tbody>
  </table>

  <p>Each <code>.npz</code> file stores the following arrays:</p>

  <table>
    <thead>
      <tr><th>Key</th><th>Shape</th><th>Description</th></tr>
    </thead>
    <tbody>
      <tr><td><code>obs_arena</code></td><td>(N, 32, 18, 7)</td><td>Arena grid</td></tr>
      <tr><td><code>obs_vector</code></td><td>(N, 23)</td><td>Vector features</td></tr>
      <tr><td><code>actions</code></td><td>(N,)</td><td>Action indices (int64)</td></tr>
      <tr><td><code>masks</code></td><td>(N, 2305)</td><td>Valid action masks</td></tr>
      <tr><td><code>timestamps</code></td><td>(N,)</td><td>Frame timestamps</td></tr>
    </tbody>
  </table>

  <h4 class="sub-title">Hierarchical BC Architecture</h4>

  <p><strong>Key insight:</strong> A flat 2305-way softmax collapses to always predicting no-op. With only ~1,600 action examples spread across 2,304 placement classes, the average is fewer than 0.7 examples per position class. The no-op class dominates with ~3,766 examples. The flat head never has enough signal to overcome the no-op prior.</p>

  <p><strong>Solution:</strong> Decompose the single 2305-way output into three specialized heads sharing a common feature extractor. This transforms an impossible 2305-class problem into three tractable sub-problems.</p>

  <p><strong>CRFeatureExtractor</strong> (192-dim output):</p>

  <ul>
    <li><strong>Arena branch (128-dim):</strong> <code>Embedding(156, 8)</code> for class IDs concatenated with 5 continuous channels produces a <code>(B, 13, 32, 18)</code> tensor. Three <code>Conv2d</code> layers (13->32->64->128) with BatchNorm, ReLU, and MaxPool reduce spatial dimensions. <code>AdaptiveAvgPool</code> yields 128 dims.</li>
    <li><strong>Vector branch (64-dim):</strong> <code>Embedding(9, 8)</code> for 4 card slots plus 19 scalar features gives a 51-dim input. Two Linear layers (51->64->64) with ReLU produce 64 dims.</li>
    <li><strong>Output:</strong> <code>cat([arena_128, vector_64]) = 192-dim</code></li>
  </ul>

  <p><strong>BCPolicy</strong> (Hierarchical 3-Head):</p>

  <table>
    <thead>
      <tr><th>Head</th><th>Output</th><th>Loss</th><th>Why It Works</th></tr>
    </thead>
    <tbody>
      <tr>
        <td>Play</td>
        <td>Binary (play vs no-op)</td>
        <td>Weighted BCE (noop=1.0, play=10.0)</td>
        <td>6:1 ratio is learnable</td>
      </tr>
      <tr>
        <td>Card</td>
        <td>4-way softmax</td>
        <td>Cross-entropy on action frames only</td>
        <td>~137 examples each</td>
      </tr>
      <tr>
        <td>Position</td>
        <td>576-way softmax</td>
        <td>Label-smoothed CE (epsilon=0.1)</td>
        <td>FiLM-conditioned per card</td>
      </tr>
    </tbody>
  </table>

  <p><strong>FiLM conditioning for position prediction:</strong> For each <code>card_id</code> (0-3), a card embedding via <code>Embedding(4, 16)</code> produces FiLM parameters through <code>gamma = Linear(16, 128)</code> and <code>beta = Linear(16, 128)</code>. Position features are then modulated as <code>gamma * features + beta</code>. This lets each card learn different spatial preferences -- for example, a defensive building should be placed near the player's towers, while an offensive troop should be placed at the bridge.</p>

  <p><strong>Total parameters: ~304K</strong></p>

  <div class="fig">
    <img src="{{ "images/final/hierarchical_vs_flat.png" | relative_url }}" alt="Hierarchical decomposition vs flat 2305-way softmax">
    <p class="caption">Hierarchical decomposition vs flat 2305-way softmax</p>
  </div>

  <h4 class="sub-title">Training Configuration</h4>

  <table>
    <thead>
      <tr><th>Parameter</th><th>Value</th></tr>
    </thead>
    <tbody>
      <tr><td>Optimizer</td><td>AdamW</td></tr>
      <tr><td>Initial LR</td><td>3e-4</td></tr>
      <tr><td>LR schedule</td><td>Cosine annealing to 1e-5</td></tr>
      <tr><td>Weight decay</td><td>1e-4</td></tr>
      <tr><td>Batch size</td><td>64</td></tr>
      <tr><td>Epochs</td><td>31 (early stopping on val F1)</td></tr>
      <tr><td>Gradient clipping</td><td>Max norm 1.0</td></tr>
      <tr><td>Dropout</td><td>0.2 (BC trunk only)</td></tr>
      <tr><td>Data augmentation</td><td>Horizontal flip (2x effective data)</td></tr>
    </tbody>
  </table>

  <p>The total loss is a sum of the three head losses:</p>

<pre><code>L_total = L_play + L_card + L_position</code></pre>

  <p>Where <code>L_play</code> uses 10:1 play-to-noop weighting, <code>L_card</code> is cross-entropy computed on action frames only, and <code>L_position</code> is per-card label-smoothed cross-entropy computed on action frames only. Gradients from all three heads flow back through the shared CRFeatureExtractor, so the feature representation is shaped by all tasks simultaneously.</p>

</div>
</section>


<!-- ============================================================ -->
<!-- APPROACH 3g: PPO REINFORCEMENT LEARNING                      -->
<!-- ============================================================ -->

<section class="reveal">
<h3 class="section-title">3g. PPO Reinforcement Learning</h3>
<div class="sp">

  <h4 class="sub-title">BC-to-PPO Transfer</h4>

  <p>The transition from offline behavior cloning to online reinforcement learning follows a targeted transfer strategy. We keep what generalizes and discard what is task-specific.</p>

  <ul>
    <li><strong>What transfers:</strong> CRFeatureExtractor weights (arena CNN + vector MLP) -- the 192-dim shared backbone that encodes game state.</li>
    <li><strong>What is discarded:</strong> All 3 BC action heads (play, card, position) are thrown away. The hierarchical decomposition was necessary for BC's small dataset but is no longer needed once the agent generates its own data.</li>
  </ul>

  <p>PPO creates new networks on top of the transferred feature extractor:</p>

  <table>
    <thead>
      <tr><th>Network</th><th>Architecture</th></tr>
    </thead>
    <tbody>
      <tr><td>Policy (pi)</td><td>[256, 128] MLP -> 2305 flat logits</td></tr>
      <tr><td>Value (vf)</td><td>[256, 128] MLP -> scalar V(s)</td></tr>
    </tbody>
  </table>

  <p><strong>Key design choice:</strong> PPO uses a flat 2305-way output rather than the hierarchical structure from BC. Online RL generates enough data through self-play that the flat action space becomes tractable -- the agent sees thousands of episodes instead of a few hundred demonstrations.</p>

  <p><strong>Temporal context via frame stacking:</strong> 3-frame stacking provides the agent with short-term temporal context. The arena observation grows from <code>(32, 18, 7)</code> to <code>(32, 18, 21)</code> and the vector observation from <code>(23,)</code> to <code>(69,)</code>. This lets the agent perceive unit movement direction and elixir change rate without recurrence.</p>

  <p>Fine-tuning proceeds in two phases to protect the transferred features:</p>

  <table>
    <thead>
      <tr><th>Phase</th><th>Feature Extractor</th><th>LR</th><th>Trainable Params</th><th>Purpose</th></tr>
    </thead>
    <tbody>
      <tr>
        <td>Phase 1 (Frozen)</td>
        <td>Frozen</td>
        <td>1e-4</td>
        <td>~148K (heads only)</td>
        <td>Calibrate RL heads without corrupting BC features</td>
      </tr>
      <tr>
        <td>Phase 2 (Unfrozen)</td>
        <td>Trainable</td>
        <td>3e-5 (10x lower)</td>
        <td>All</td>
        <td>End-to-end adaptation</td>
      </tr>
    </tbody>
  </table>

  <h4 class="sub-title">Live Game Environment</h4>

  <p><strong>ClashRoyaleEnv</strong> is a Gymnasium-compatible wrapper around the live commercial game. There is no simulator -- every episode is a real match against a real opponent found through the game's matchmaking system.</p>

  <ul>
    <li><strong>Step cycle:</strong> Capture -> Detect -> Act -> Perceive -> Reward</li>
    <li><strong>Step budget:</strong> ~237ms actual, 500ms maximum</li>
    <li><strong>Episode lifecycle:</strong> Wait for match -> Game start detected -> Step loop -> Game end detection</li>
    <li><strong>Action masking:</strong> <code>MaskablePPO</code> from <code>sb3-contrib</code> handles invalid action masking, preventing the agent from selecting cards it does not have or placing units in illegal positions.</li>
  </ul>

  <table>
    <thead>
      <tr><th>Parameter</th><th>Value</th></tr>
    </thead>
    <tbody>
      <tr><td>Algorithm</td><td>MaskablePPO (sb3-contrib)</td></tr>
      <tr><td>LR schedule</td><td>Cosine decay 1e-4 -> 1e-5</td></tr>
      <tr><td>Clip range</td><td>0.1</td></tr>
      <tr><td>n_steps</td><td>700 (approx. one full match)</td></tr>
      <tr><td>Batch size</td><td>64</td></tr>
      <tr><td>n_epochs</td><td>10</td></tr>
      <tr><td>Gamma</td><td>0.99</td></tr>
      <tr><td>GAE lambda</td><td>0.95</td></tr>
      <tr><td>Entropy coefficient</td><td>0.02 annealed to 0.005</td></tr>
      <tr><td>Max grad norm</td><td>0.5</td></tr>
    </tbody>
  </table>

  <h4 class="sub-title">Reward Engineering: 6 Iterations</h4>

  <p>Designing a reward function for a live strategy game is an iterative process. Each fix addressed one failure mode but often created a new one. We went through 6 reward iterations, each a mini-experiment with a hypothesis, result, and lesson learned:</p>

  <table>
    <thead>
      <tr><th>Iter</th><th>Change</th><th>Result</th></tr>
    </thead>
    <tbody>
      <tr>
        <td>1</td>
        <td>Win/loss only</td>
        <td>Too sparse -- agent learns nothing</td>
      </tr>
      <tr>
        <td>2</td>
        <td>+ Survival bonus (+0.02/step)</td>
        <td>Passive -- afraid to play cards</td>
      </tr>
      <tr>
        <td>3</td>
        <td>+ Elixir waste penalty (-0.1 at 9.5+)</td>
        <td>Plays randomly to spend elixir</td>
      </tr>
      <tr>
        <td>4</td>
        <td>+ Unit advantage (0.03 x delta)</td>
        <td>Spams cheap troops, ignores positioning</td>
      </tr>
      <tr>
        <td>5</td>
        <td>+ Defensive placement (+0.15 near enemies)</td>
        <td>FIRST WINS -- session 16: 10%</td>
      </tr>
      <tr>
        <td>6</td>
        <td>+ Low-elixir noop bonus (+0.01 below 3)</td>
        <td>Better resource management -- session 19: 27%</td>
      </tr>
    </tbody>
  </table>

  <p><strong>Terminal rewards:</strong> Win/Loss +/-3.0, Crowns +/-1.0 (all scaled by 0.1). These provide the anchor signal that step-level shaping rewards orient toward.</p>

  <div class="two-col">
    <div class="fig">
      <img src="{{ "images/final/ppo_reward_evolution.png" | relative_url }}" alt="Reward evolution across 6 iterations and 106 episodes">
      <p class="caption">Reward evolution across 6 iterations and 106 episodes.</p>
    </div>
    <div class="fig">
      <img src="{{ "images/final/reward_components_breakdown.png" | relative_url }}" alt="Breakdown of reward components contributing to total reward signal">
      <p class="caption">Reward component breakdown showing the relative contribution of each shaping signal.</p>
    </div>
  </div>

</div>
</section>

<hr class="divider">


<!-- ============================================================ -->
<!-- EVALUATION                                                     -->
<!-- ============================================================ -->

<section class="reveal">
  <h2 class="section-title" style="background: linear-gradient(135deg, var(--cr-sky-mid), var(--cr-blue-dark)); padding: var(--space-md) var(--space-lg); border-radius: var(--border-radius-lg);">Evaluation</h2>
</section>


<!-- 4a. Object Detection Performance -->
<section class="reveal">
<div class="sp">
  <h3>4a. Object Detection Performance</h3>

  <p>Our dual YOLOv8m detector was evaluated on 1,388 human-annotated real gameplay images from the KataCR validation set -- images the model never saw during training.</p>

  <div class="metric-row">
    <div class="metric-box gold"><span class="num">0.804</span><span class="label">mAP50</span></div>
    <div class="metric-box"><span class="num">0.567</span><span class="label">mAP50-95</span></div>
    <div class="metric-box"><span class="num">0.822</span><span class="label">Precision</span></div>
    <div class="metric-box"><span class="num">0.771</span><span class="label">Recall</span></div>
    <div class="metric-box"><span class="num">0.795</span><span class="label">F1 (conf=0.765)</span></div>
    <div class="metric-box"><span class="num">15.3</span><span class="label">FPS</span></div>
  </div>

  <h4 class="sub-title">Training Progression</h4>

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

  <p>mAP50 plateaued around 0.80 after epoch 10, indicating that the coarse detection ability was learned quickly. mAP50-95 -- which requires tighter bounding box precision -- kept slowly improving through epoch 50, reflecting gradual refinement of localization accuracy.</p>

  <div class="fig">
    <img src="{{ "images/final/b1_map_progression.png" | relative_url }}" alt="mAP50 and mAP50-95 progression across 50 training epochs">
    <p class="caption">mAP50 and mAP50-95 progression across 50 training epochs.</p>
  </div>

  <h4 class="sub-title">Model Comparison</h4>

  <p>We evaluated three model configurations. The YOLOv8n baseline was trained on real images with validation data leakage, making its 0.944 mAP50 misleading. Our dual YOLOv8m achieves an honest 0.804 on a properly held-out validation set, with zero hallucinations and substantially better coverage.</p>

  <table>
    <thead>
      <tr><th>Metric</th><th>YOLOv8n (old)</th><th>YOLOv8s (single)</th><th>Dual YOLOv8m (final)</th></tr>
    </thead>
    <tbody>
      <tr><td>Training data</td><td>Real (val leak)</td><td>Synthetic only</td><td>Synthetic only</td></tr>
      <tr><td>Parameters</td><td>3.2M</td><td>11.2M</td><td>~51.8M (2x 25.9M)</td></tr>
      <tr><td>mAP50</td><td>0.944 (misleading)</td><td>0.756</td><td><strong>0.804</strong> (honest)</td></tr>
      <tr><td>Detections/frame</td><td>20.3</td><td>24.1</td><td><strong>30.2</strong> (+49%)</td></tr>
      <tr><td>Hallucinations</td><td>50</td><td>12</td><td><strong>0</strong></td></tr>
      <tr><td>Unique classes seen</td><td>57</td><td>63</td><td><strong>85</strong> (+49%)</td></tr>
    </tbody>
  </table>

  <div class="fig">
    <img src="{{ "images/final/b4_model_comparison.png" | relative_url }}" alt="Model comparison: YOLOv8n vs single YOLOv8s vs dual YOLOv8m">
    <p class="caption">Model comparison: YOLOv8n vs single YOLOv8s vs dual YOLOv8m.</p>
  </div>

  <h4 class="sub-title">AP50 Tier Distribution</h4>

  <p>Breaking down per-class AP50 across 155 detection classes shows that nearly half achieve strong performance, while a tail of 14 classes remains below 0.40 -- mostly transient spell effects and rare units.</p>

  <table>
    <thead>
      <tr><th>AP50 Tier</th><th>Classes</th><th>Share</th></tr>
    </thead>
    <tbody>
      <tr><td>Strong (>=0.90)</td><td>73</td><td>47%</td></tr>
      <tr><td>Adequate (0.70-0.89)</td><td>32</td><td>21%</td></tr>
      <tr><td>Weak (0.40-0.69)</td><td>20</td><td>13%</td></tr>
      <tr><td>Failing (<0.40)</td><td>14</td><td>9%</td></tr>
    </tbody>
  </table>

  <div class="fig">
    <img src="{{ "images/final/b3_ap50_distribution.png" | relative_url }}" alt="AP50 distribution across 155 detection classes">
    <p class="caption">AP50 distribution across 155 detection classes.</p>
  </div>

  <h4 class="sub-title">Domain Gap Analysis</h4>

  <p>The primary bottleneck is classification, not localization. Synthetic sprites lack the animation frames, lighting variation, and particle effects present in real gameplay rendering. The train-to-validation loss ratios quantify this gap:</p>

  <table>
    <thead>
      <tr><th>Loss Component</th><th>Train/Val Ratio</th><th>Interpretation</th></tr>
    </thead>
    <tbody>
      <tr><td>Box loss</td><td>3.9x</td><td>Moderate gap</td></tr>
      <tr><td>Classification</td><td><strong>4.7x</strong></td><td>Primary bottleneck</td></tr>
      <tr><td>DFL (distribution)</td><td>2.0x</td><td>Smallest gap</td></tr>
    </tbody>
  </table>

  <p>mAP50 progression tells the same story: 0.675 (epoch 1) to 0.797 (epoch 10) to 0.804 (epoch 50). The plateau after epoch 10 means additional training on synthetic data cannot close the domain gap -- real-frame fine-tuning would be needed to push beyond 0.80.</p>

  <div class="fig">
    <img src="{{ "images/final/b5_domain_gap.png" | relative_url }}" alt="Domain gap analysis: train vs validation loss divergence">
    <p class="caption">Domain gap analysis: train vs validation loss divergence.</p>
  </div>

</div>
</section>


<!-- 4b. Behavior Cloning Results -->
<section class="reveal">
<div class="sp">
  <h3>4b. Behavior Cloning Results</h3>

  <p>Behavior cloning was trained on 40 expert-played matches (5,366 frames, ~1,600 action frames) to provide a warm-start policy for PPO fine-tuning.</p>

  <div class="metric-row">
    <div class="metric-box gold"><span class="num">0.384</span><span class="label">Best F1</span></div>
    <div class="metric-box"><span class="num">99%</span><span class="label">Action Recall</span></div>
    <div class="metric-box"><span class="num">36.7%</span><span class="label">Card Accuracy</span></div>
    <div class="metric-box"><span class="num">79%</span><span class="label">Position Loss Share</span></div>
  </div>

  <h4 class="sub-title">Training Progression</h4>

  <table>
    <thead>
      <tr><th>Metric</th><th>Epoch 1</th><th>Best</th><th>Epoch 32</th></tr>
    </thead>
    <tbody>
      <tr><td>Train loss</td><td>7.735</td><td>-</td><td>4.504</td></tr>
      <tr><td>Val loss</td><td>7.132</td><td>6.913 (ep 5)</td><td>10.630</td></tr>
      <tr><td>Action F1</td><td>0.280</td><td><strong>0.384</strong> (ep 6)</td><td>0.289</td></tr>
      <tr><td>Action Recall</td><td>0.991</td><td>1.000 (ep 12)</td><td>0.897</td></tr>
      <tr><td>Card Accuracy</td><td>0.259</td><td><strong>0.367</strong> (ep 13)</td><td>0.253</td></tr>
    </tbody>
  </table>

  <p>The play head reaches near-perfect recall quickly -- the model learns <em>when</em> to play cards within the first few epochs. Card accuracy exceeds the 25% random baseline (4-class uniform), showing meaningful card selection learning. However, position prediction is the clear bottleneck: with 576 possible cells and only ~1,600 action examples, the model sees fewer than 3 examples per cell on average. Position loss accounts for 79% of total loss. Validation loss diverges after epoch 5, showing clear overfitting on the small dataset. The best F1 of 0.384 at epoch 6 was used for model selection.</p>

  <div class="two-col">
    <div class="fig">
      <img src="{{ "images/final/b8_bc_training_curves.png" | relative_url }}" alt="BC training and validation loss curves">
      <p class="caption">BC training and validation loss curves.</p>
    </div>
    <div class="fig">
      <img src="{{ "images/final/bc_loss_decomposition.png" | relative_url }}" alt="Loss decomposition: position dominates at 79%">
      <p class="caption">Loss decomposition: position dominates at 79%.</p>
    </div>
  </div>

  <div class="fig">
    <img src="{{ "images/final/b9_bc_head_performance.png" | relative_url }}" alt="Per-head performance: play, card, and position head metrics across training">
    <p class="caption">Per-head performance: play head converges quickly, card head exceeds random baseline, position head remains the bottleneck.</p>
  </div>

</div>
</section>


<!-- 4c. PPO Training Results -->
<section class="reveal">
<div class="sp">
  <h3>4c. PPO Training Results</h3>

  <p>After warm-starting from the behavior cloning policy, we fine-tuned with MaskablePPO across 19 training sessions on live matches against real human opponents. This is the core contribution of the project -- learning to win real games from live experience.</p>

  <div class="metric-row">
    <div class="metric-box gold"><span class="num">30%</span><span class="label">Win Rate</span></div>
    <div class="metric-box"><span class="num">106</span><span class="label">Live Episodes</span></div>
    <div class="metric-box gold"><span class="num">5</span><span class="label">Wins</span></div>
    <div class="metric-box"><span class="num">19</span><span class="label">Training Sessions</span></div>
  </div>

  <h4 class="sub-title">Training Progression</h4>

  <table>
    <thead>
      <tr><th>Phase</th><th>Episodes</th><th>Win Rate</th><th>Avg Reward</th></tr>
    </thead>
    <tbody>
      <tr><td>Learning fundamentals</td><td>1-74</td><td><strong>0%</strong></td><td>-13 (worst)</td></tr>
      <tr><td>First wins (session 16)</td><td>75-83</td><td><strong>10%</strong></td><td>first 2 wins</td></tr>
      <tr><td>Late training (session 19)</td><td>95-106</td><td><strong>25%</strong></td><td><strong>-2.6</strong></td></tr>
      <tr><td><strong>Final eval (10 games)</strong></td><td>--</td><td><strong>30%</strong></td><td>--</td></tr>
    </tbody>
  </table>

  <p>Overall training record: <strong>5W / 80L / 3D / 18 unknown</strong> (unknowns are interrupted matches). The agent spent 74 episodes losing every game while learning fundamental behaviors, then began winning in session 16. By session 19, average reward improved from -13 to -2.6, indicating much closer games even in losses. The final evaluation -- 10 fresh games with the best checkpoint -- yielded a 30% win rate against real human opponents.</p>

  <p><strong>Note on sample size:</strong> With n=10 evaluation games, the 30% win rate has substantial uncertainty. A larger evaluation set would be needed to pin down the true win rate precisely. However, the trend is robust: the agent improved from 0% (episodes 1-74) to 10% (session 16) to 27% (session 19) to 30% (final eval) -- consistent upward progression across independent measurement points.</p>

  <div class="fig">
    <img src="{{ "images/final/ppo_training_progression.png" | relative_url }}" alt="PPO training progression across 106 live episodes">
    <p class="caption">PPO training progression across 106 live episodes.</p>
  </div>

  <h4 class="sub-title">Card Usage</h4>

  <p>Across 106 episodes, the agent made 2,815 total card plays. The distribution reveals sensible elixir management:</p>

  <table>
    <thead>
      <tr><th>Card</th><th>Plays</th><th>%</th></tr>
    </thead>
    <tbody>
      <tr><td>electro-spirit</td><td>404</td><td>14%</td></tr>
      <tr><td>flying-machine</td><td>399</td><td>14%</td></tr>
      <tr><td>goblin-cage</td><td>383</td><td>14%</td></tr>
      <tr><td>zappies</td><td>383</td><td>14%</td></tr>
      <tr><td>barbarian-barrel</td><td>344</td><td>12%</td></tr>
      <tr><td>arrows</td><td>304</td><td>11%</td></tr>
      <tr><td>royal-hogs</td><td>142</td><td>5%</td></tr>
      <tr><td>royal-recruits</td><td>58</td><td>2%</td></tr>
    </tbody>
  </table>

  <p>The agent uses all 8 cards with no dead cards in the deck. Cheap cards are preferred, reflecting efficient elixir management. Expensive cards like royal-recruits (7 elixir) are played sparingly. Lane balance is nearly even at 50% left / 50% right. Average elixir at play time: 5.1.</p>

  <div class="fig">
    <img src="{{ "images/final/ppo_card_usage.png" | relative_url }}" alt="Card usage distribution across 106 training episodes">
    <p class="caption">Card usage distribution across 106 training episodes.</p>
  </div>

  <h4 class="sub-title">Emergent Behaviors</h4>

  <p>After 106 episodes, the agent developed five measurable behaviors without explicit programming -- these emerged purely from the reward signal and live experience:</p>

  <table>
    <thead>
      <tr><th>Behavior</th><th>Evidence</th></tr>
    </thead>
    <tbody>
      <tr><td>Balanced elixir spending</td><td>Avg 5.1 elixir at play time</td></tr>
      <tr><td>Defensive placement</td><td>Places troops near enemy units (+0.15 shaped reward)</td></tr>
      <tr><td>Card variety</td><td>All 8 cards used, weighted by cost</td></tr>
      <tr><td>Lane balance</td><td>50/50 left-right split -- no tunnel vision</td></tr>
      <tr><td>Efficient stepping</td><td>Avg 21.6 cards/game, noop ratio only 10.1%</td></tr>
    </tbody>
  </table>

  <p>The agent learned to spend resources when it has them, defend when threatened, and distribute pressure across both lanes -- all from reward signals and live experience. That said, the 50/50 lane split and balanced card usage could partly reflect uniform placement rather than deliberate strategic choices -- further analysis of game-state-conditioned placement patterns would be needed to fully distinguish strategic from random behavior.</p>

  <div class="fig">
    <img src="{{ "images/final/ppo_final_eval_comparison.png" | relative_url }}" alt="Final evaluation: 30% win rate in 10-game test">
    <p class="caption">Final evaluation: 30% win rate in 10-game test.</p>
  </div>

  <h4 class="sub-title">Qualitative Analysis: Why the Agent Loses</h4>

  <p>From watching replays of losses, we identified three dominant failure modes that account for the majority of the 70% loss rate:</p>

  <p><strong>1. Positioning errors.</strong> The agent places defensive units too far from incoming enemy pushes. For example, when an enemy deploys a Hog Rider at the bridge, the agent may place Zappies at the back of the arena rather than near the threatened Princess Tower. This stems from the BC position head bottleneck -- with fewer than 3 examples per grid cell, the model learned coarse spatial priors (deploy near your side) but not precise reactive placement. A human player would place the counter-troop directly in the Hog Rider's path.</p>

  <p><strong>2. Inability to read opponent pushes.</strong> The agent has no temporal memory beyond the 3-frame stack (~1.5 seconds at 2 FPS). It cannot track a slow-building push accumulating on one lane. By the time the push crosses the bridge and becomes visible in the current frame, it is often too late to mount an effective defense. Human players read the opponent's card cycle and elixir spending patterns to anticipate pushes before they materialize.</p>

  <p><strong>3. Spell detection failures.</strong> Arrows (AP50=0.190) and Zap (AP50=0.245) are consistently missed by the detector due to their transient visual effects. When the opponent uses spells to clear the agent's troops, the agent does not perceive the loss and may continue investing elixir into a lane that has already been countered. This creates cascading waste that human opponents exploit.</p>

  <p>In wins, the agent typically succeeds by applying sustained dual-lane pressure with cheap troops (Electro Spirit, Barbarian Barrel, Goblin Cage), forcing the opponent to defend both lanes simultaneously. The agent's fast reaction time (~237ms) and willingness to always spend elixir above 3 creates constant board presence that overwhelms opponents who over-commit to one lane.</p>

</div>
</section>


<!-- 4d. System Performance -->
<section class="reveal">
<div class="sp">
  <h3>4d. System Performance</h3>

  <h4 class="sub-title">Latency Breakdown</h4>

  <table>
    <thead>
      <tr><th>Component</th><th>Time</th></tr>
    </thead>
    <tbody>
      <tr><td>Screen capture (mss)</td><td>~5ms</td></tr>
      <tr><td>YOLO detection</td><td>~150ms</td></tr>
      <tr><td>OCR (PaddleOCR)</td><td>~50ms</td></tr>
      <tr><td>Card classification</td><td>~10ms</td></tr>
      <tr><td>State encoding</td><td>~2ms</td></tr>
      <tr><td>Policy inference</td><td>~5ms</td></tr>
      <tr><td>Action execution (PyAutoGUI)</td><td>~15ms</td></tr>
      <tr><td><strong>Total</strong></td><td><strong>~237ms</strong></td></tr>
    </tbody>
  </table>

  <p>YOLO detection dominates at 63% of total latency. The entire pipeline fits well within the 500ms frame budget, leaving 263ms of headroom. This means the agent can comfortably operate at 2-4 FPS, sufficient for the game's real-time decision cadence.</p>

  <div class="fig">
    <img src="{{ "images/final/system_latency.png" | relative_url }}" alt="System latency breakdown per frame">
    <p class="caption">System latency breakdown per frame.</p>
  </div>

  <h4 class="sub-title">Test Coverage</h4>

  <table>
    <thead>
      <tr><th>Module</th><th>Tests</th></tr>
    </thead>
    <tbody>
      <tr><td>State Encoder</td><td>42</td></tr>
      <tr><td>Action Builder</td><td>46</td></tr>
      <tr><td>Dataset Builder</td><td>24</td></tr>
      <tr><td>PPO Module</td><td>41</td></tr>
      <tr><td><strong>Total</strong></td><td><strong>153</strong></td></tr>
    </tbody>
  </table>

</div>
</section>


<!-- 4e. Challenges and Limitations -->
<section class="reveal">
<div class="sp">
  <h3>4e. Challenges and Limitations</h3>

  <div class="challenge high">
    <h4>Synthetic-to-Real Domain Gap</h4>
    <p>The classification loss train/val ratio of 4.7x is the largest gap across all loss components. Synthetic sprites lack animation frames, dynamic lighting, and particle effects present in real gameplay rendering. mAP50 plateaus at 0.804 after epoch 10, and no amount of additional synthetic training can close this gap. Pushing detection accuracy further requires fine-tuning on labeled real gameplay frames.</p>
  </div>

  <div class="challenge high">
    <h4>Position Head Bottleneck</h4>
    <p>The action space has 576 placement cells, but only ~1,600 action examples exist in the BC dataset -- fewer than 3 examples per cell on average. Position loss accounts for 79% of total BC loss. The model cannot learn fine-grained placement strategy from this data alone. Addressing this requires 20-40 more recorded matches for BC data, combined with PPO fine-tuning to discover effective placements through trial and error.</p>
  </div>

  <div class="challenge med">
    <h4>No Temporal Context in BC</h4>
    <p>The behavior cloning model operates on single-frame state with no memory of previous frames. It cannot track unit trajectories, estimate opponent elixir based on recent plays, or learn timing-dependent strategies. PPO's 3-frame stacking partially addresses this by giving the policy a short history window, but a recurrent or transformer-based architecture would be needed for full temporal reasoning.</p>
  </div>

  <div class="challenge med">
    <h4>No Self-Play Environment</h4>
    <p>All training happens against real human opponents via matchmaking. This means opponent skill varies unpredictably, difficulty cannot be controlled or gradually increased, and training throughput is limited to real-time match speed. A self-play environment would enable curriculum learning and massively parallel training, but requires a game simulator that does not exist for Clash Royale.</p>
  </div>

  <div class="challenge low">
    <h4>Small Data Regime</h4>
    <p>Our BC dataset contains 40 matches (5,366 frames) and PPO training ran for 106 episodes. For comparison, AlphaStar trained on millions of games and OpenAI Five on billions of frames. While our results demonstrate that the approach works, significantly more data and training time would likely improve performance substantially.</p>
  </div>

  <div class="challenge low">
    <h4>Spell Detection Weakness</h4>
    <p>Transient visual effects are inherently difficult to capture from single frames. Arrows achieves only 0.190 AP50 and Zap only 0.245 AP50. These spells appear as brief animated effects that look nothing like their sprite cutouts. Addressing this may require temporal detection approaches or specialized spell-recognition models.</p>
  </div>

</div>
</section>

<hr class="divider">


<!-- ============================================================ -->
<!-- RESOURCES USED                                                -->
<!-- ============================================================ -->

<section class="reveal">
  <h2 class="section-title" style="background: linear-gradient(135deg, var(--cr-sky-mid), var(--cr-blue-dark)); padding: var(--space-md) var(--space-lg); border-radius: var(--border-radius-lg);">Resources Used</h2>
</section>

<section class="reveal">
<div class="sp">

  <h4 class="sub-title">Core Libraries</h4>

  <ul>
    <li><strong>Ultralytics YOLOv8</strong> -- object detection framework</li>
    <li><strong>PaddleOCR</strong> -- optical character recognition for game text</li>
    <li><strong>Stable Baselines 3 + SB3-Contrib (MaskablePPO)</strong> -- reinforcement learning</li>
    <li><strong>PyTorch</strong> -- deep learning framework</li>
    <li><strong>OpenCV</strong> -- image processing</li>
    <li><strong>mss</strong> -- screen capture</li>
    <li><strong>PyAutoGUI</strong> -- mouse automation</li>
    <li><strong>pynput</strong> -- OS-level input logging</li>
    <li><strong>Flask</strong> -- web server for Quick-Seg tool</li>
    <li><strong>NumPy, SciPy</strong> -- numerical computation</li>
  </ul>

  <h4 class="sub-title">Datasets and External Code</h4>

  <ul>
    <li><strong>KataCR</strong> (MIT License) -- 155-class detection schema, synthetic data generation approach, sprite cutouts. <a href="https://github.com/wty-yy/KataCR">github.com/wty-yy/KataCR</a></li>
    <li><strong>Clash Royale Detection Dataset</strong> -- 6,939 human-annotated real images. We used 1,388 as a held-out validation set. <a href="https://github.com/wty-yy/Clash-Royale-Detection-Dataset">github.com/wty-yy/Clash-Royale-Detection-Dataset</a>. We forked this and added 383 custom ally sprite cutouts.</li>
    <li><strong>ClashRoyaleBuildABot</strong> -- design reference for game automation approach. <a href="https://github.com/Pbatch/ClashRoyaleBuildABot">github.com/Pbatch/ClashRoyaleBuildABot</a></li>
  </ul>

  <h4 class="sub-title">Platform</h4>

  <ul>
    <li><strong>Google Play Games</strong> (PC emulator) -- native Windows rendering, consistent screen dimensions, no Android device variance</li>
  </ul>

  <h4 class="sub-title">AI Tool Usage</h4>

  <p>Claude (Anthropic) was used for:</p>

  <ul>
    <li>Boilerplate code scaffolding and documentation drafting</li>
    <li>Debugging assistance and code review</li>
    <li>Website design and report formatting</li>
  </ul>

  <p>All core technical work -- model architecture design, training pipeline implementation, reward engineering, data collection, and evaluation -- was done by the team. AI-generated code was reviewed and modified before integration.</p>

</div>
</section>

<hr class="divider">
