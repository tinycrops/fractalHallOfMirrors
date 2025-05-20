# ─────────────────────────────────────────────────────────────────────────────
#  fractal_meta_bot.py         Canonical-space FPS bot + Fractal Ops Manifold
# ─────────────────────────────────────────────────────────────────────────────
"""
Demonstrates a self-improving agent inside a visually "infinite" FPS map.
Each opponent encounter is a TIP (Task-Instance-Projection) of the Fractal
Operations Manifold.  What a TIP learns (spray-pattern, flanking path, …)
is broadcast by the Universal-Learning-Propagator to every future encounter
of the same Fractal-Pattern ("combat/shotgun", "combat/rocket", …).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing    import Dict, List, Tuple, Any, Optional
import heapq, math, time, uuid, random, threading, queue, collections
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import to_rgba
# ════════════════════════════════════════════════════════════════════════════
#  Basic vector helpers
# ════════════════════════════════════════════════════════════════════════════
V3 = Tuple[float, float, float]
def v_add(a:V3,b:V3)->V3: return (a[0]+b[0],a[1]+b[1],a[2]+b[2])
def v_sub(a:V3,b:V3)->V3: return (a[0]-b[0],a[1]-b[1],a[2]-b[2])
def v_len(a:V3)->float   : return math.sqrt(a[0]**2+a[1]**2+a[2]**2)
def v_dist(a:V3,b:V3)->float: return v_len(v_sub(a,b))
# ════════════════════════════════════════════════════════════════════════════
#  Game-world data structures (canonical map)
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class SelfState:
    pos:V3=(0,0,0); yaw:float=0; health:int=100; ammo:int=50; weapon:str="AR"
    vel:V3=(0,0,0)
@dataclass
class Opponent:
    id:str; pos:V3; yaw:float; weapon:str; last_seen:float
    confidence:float=1.0
@dataclass
class CanonicalLevel:
    nav:Dict[V3,List[V3]]            # adjacency list
    def path(self,start:V3,goal:V3)->List[V3]:
        if start not in self.nav or goal not in self.nav: return []
        open:List[Tuple[float,V3]]=[(0,start)]
        came:Dict[V3,Optional[V3]]={start:None}; g={start:0}
        while open:
            _,cur=heapq.heappop(open)
            if cur==goal:
                path=[]
                while cur: path.append(cur); cur=came[cur]
                return path[::-1]
            for nb in self.nav[cur]:
                t=g[cur]+v_dist(cur,nb)
                if t<g.get(nb,1e9):
                    came[nb]=cur; g[nb]=t
                    heapq.heappush(open,(t+v_dist(nb,goal),nb))
        return []
# ════════════════════════════════════════════════════════════════════════════
#  Perception module (canonicalises the infinite visuals)
# ════════════════════════════════════════════════════════════════════════════
class Perception:
    def __init__(self,engine,kb): self.eg=engine; self.kb=kb
    def tick(self):
        raw_me=self.eg.get_self()
        self.kb.self.pos=self._canon(raw_me["pos"])
        self.kb.self.yaw=raw_me["yaw"]; self.kb.self.health=raw_me["hp"]
        now=time.time()
        for ent in self.eg.visible():
            if ent["type"]!="opponent": continue
            cid=ent["id"]; cp=self._canon(ent["pos"])
            op=self.kb.opp.get(cid) or Opponent(cid,cp,ent["yaw"],ent["weapon"],now)
            op.pos=cp; op.yaw=ent["yaw"]; op.weapon=ent["weapon"]; op.last_seen=now
            self.kb.opp[cid]=op
        # decay confidence
        for o in self.kb.opp.values():
            if now-o.last_seen>0.5: o.confidence*=0.95
    def _canon(self,p:V3)->V3:                   # strip layer-index on z
        x,y,z=p; h=self.eg.layer_height(); return (x,y,z%h)
    def los(self,target:V3)->bool: return self.eg.los(self.kb.self.pos,target)
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class KnowledgeBase:
    level:CanonicalLevel
    self:SelfState=field(default_factory=SelfState)
    opp:Dict[str,Opponent]=field(default_factory=dict)
# ════════════════════════════════════════════════════════════════════════════
#  ─── FOM layer ─────────────────────────────────────────────────────────────
#  FractalPattern -> TIPs -> Universal Learning
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class FractalPattern:
    key:str; desc:str; stats:Dict[str,float]=field(default_factory=lambda:collections.defaultdict(float))
    # learned knobs every TIP can reuse
    aim_offset:float=0.0          # simplistic example: systematic recoil error
    flank_bias:float=1.0          # multiplies path cost for exposed edges
    def integrate(self,exp:Dict[str,Any]):
        self.stats["seen"]+=1
        if exp.get("hit"):                       # recoil correction
            err=exp["aim_error"]
            self.aim_offset=0.8*self.aim_offset+0.2*err
        if exp.get("took_damage")==0:            # good flank
            self.flank_bias*=0.95
# ════════════════════════════════════════════════════════════════════════════
class CoreAgent:
    def __init__(self,kb,perception,action):
        self.kb=kb; self.per=perception; self.act=action
        self.patterns:Dict[str,FractalPattern]={}
        self.queue:queue.Queue[Tuple[str,Any]] = queue.Queue()
        # Combat metrics tracking
        self.metrics = {
            'total_shots': 0,
            'hits': 0,
            'misses': 0,
            'damage_taken': 0,
            'shots_by_pattern': collections.defaultdict(int),
            'hits_by_pattern': collections.defaultdict(int),
            'hit_history': []  # List of (timestamp, pattern_key, hit/miss)
        }
        threading.Thread(target=self._dispatcher,daemon=True).start()
    # ―― Pattern recognition (very crude baseline) ―――――――――――――――――――――――
    def _key_for(self,op:Opponent)->str: return f"combat/{op.weapon}"
    # ―― submit encounter as a new TIP ――――――――――――――――――――――――――――――――
    def encounter(self,op:Opponent):
        key=self._key_for(op)
        fp=self.patterns.get(key) or FractalPattern(key,f"auto {key}")
        self.patterns[key]=fp
        self.queue.put(("combat",dict(op_id=op.id,pattern=fp)))
    # ―― ULP: called by TIP threads ―――――――――――――――――――――――――――――――――――
    def ulp(self,fp:FractalPattern,exp:Dict[str,Any]):
        fp.integrate(exp)
    # ―― dispatcher launches TIPs ――――――――――――――――――――――――――――――――――――
    def _dispatcher(self):
        while True:
            typ,args=self.queue.get()
            if typ=="combat":
                CombatTIP(core=self,**args).start()
# ════════════════════════════════════════════════════════════════════════════
class CombatTIP(threading.Thread):
    def __init__(self,core:CoreAgent,op_id:str,pattern:FractalPattern):
        super().__init__(daemon=True)
        self.core=core; self.kb=core.kb; self.per=core.per; self.act=core.act
        self.op_id=op_id; self.fp=pattern
        self.current_path = []  # Store the current path for visualization
        self.aim_position = None  # Store where we're aiming
        self.shot_fired = False
    # ―― main routine ―――――――――――――――――――――――――――――――――――――――――――――――
    def run(self):
        op=self.kb.opp.get(self.op_id)
        if not op: return
        # 1. Plan path with pattern-aware flank bias
        path=self._path_to(op.pos, self.fp.flank_bias)
        self.current_path = path  # Store for visualization
        # 2. Walk first step
        if len(path)>1: self.act.move_towards(path[1])
        # 3. Aim using pattern's learned recoil offset
        aim_error=random.gauss(self.fp.aim_offset,0.02)     # simulate inaccuracy
        aim_pos=(op.pos[0]+aim_error, op.pos[1], op.pos[2])
        self.aim_position = aim_pos  # Store for visualization
        
        # 4. Shoot if we have line of sight
        if self.per.los(aim_pos): 
            self.act.look_at(aim_pos)
            self.act.shoot()
            self.shot_fired = True
            
            # Track shot metrics
            hit = random.random() < 0.6  # pretend 60% shot landed
            self.core.metrics['total_shots'] += 1
            self.core.metrics['shots_by_pattern'][self.fp.key] += 1
            
            if hit:
                self.core.metrics['hits'] += 1
                self.core.metrics['hits_by_pattern'][self.fp.key] += 1
            else:
                self.core.metrics['misses'] += 1
                
            # Record hit history for visualization
            self.core.metrics['hit_history'].append((time.time(), self.fp.key, hit))
            
            # 5. Build experience package
            exp=dict(hit=hit,
                     aim_error=aim_error,
                     took_damage=random.randint(0,1))
                    
            # Track damage taken
            if exp['took_damage'] > 0:
                self.core.metrics['damage_taken'] += exp['took_damage']
                
            # 6. ULP
            self.core.ulp(self.fp,exp)
    # ―― biased path-finding ――――――――――――――――――――――――――――――――――――――――
    def _path_to(self,goal:V3,bias:float)->List[V3]:
        lvl=self.kb.level
        # modify edge weights on exposed tiles (dummy heuristic)
        original=lvl.nav
        # For brevity, reuse nav as-is; in a real impl you'd multiply weights
        return lvl.path(self.kb.self.pos,goal)
# ════════════════════════════════════════════════════════════════════════════
#  Low-level action module (wraps engine commands)
# ════════════════════════════════════════════════════════════════════════════
class Action:
    def __init__(self,engine,kb): self.eg=engine; self.kb=kb
    def move_towards(self,pos:V3): self.eg.move(v_sub(pos,self.kb.self.pos))
    def look_at(self,pos:V3):      self.eg.turn(pos)
    def shoot(self):               self.eg.shoot()
# ════════════════════════════════════════════════════════════════════════════
#  Dummy engine so you can run the file and watch cross-TIP learning happen
# ════════════════════════════════════════════════════════════════════════════
class DummyEngine:
    def __init__(self):
        # Tiny square map with wrapping edge to emulate fractal portal
        a=(0,0,0); b=(1,0,0); c=(1,1,0); d=(0,1,0)
        self.nav={a:[b,d], b:[a,c], c:[b,d], d:[c,a]}
        self._tick=0
    def get_nav(self): return self.nav
    def get_self(self):
        return dict(pos=(0,0,0),yaw=0,hp=100)
    def visible(self):
        # Generate one opponent every second tick, alternating weapons
        self._tick+=1
        if self._tick%2==0:
            wid="shotgun" if self._tick%4==0 else "rocket"
            return [dict(type="opponent",id=f"op{self._tick}",pos=(1,1,0),
                         yaw=180,weapon=wid)]
        return []
    def layer_height(self): return 5.0
    def los(self,a:V3,b:V3): return True
    #  Commands --------------------------------------------------------------
    def move(self,vec):   pass
    def turn(self,pos):   pass
    def shoot(self):      pass
# ════════════════════════════════════════════════════════════════════════════
#  Visualization components
# ════════════════════════════════════════════════════════════════════════════
class FractalVisualizer:
    def __init__(self, kb: KnowledgeBase, meta: CoreAgent):
        self.kb = kb
        self.meta = meta
        self.per = meta.per  # Add perception reference
        
        # Data storage for visualization
        self.history = {
            'time': [],
            'patterns': {},
            'positions': [],
            'accuracy': []
        }
        
        # Setup plots
        self.setup_plots()
        
        # Update frequency
        self.last_update = time.time()
        self.update_interval = 0.5  # seconds
        
    def setup_plots(self):
        plt.ion()  # Enable interactive mode
        self.fig = plt.figure(figsize=(15, 10))
        
        # Main layout with 2x2 grid
        gs = self.fig.add_gridspec(2, 2)
        
        # World map visualization
        self.ax_world = self.fig.add_subplot(gs[0, 0])
        self.ax_world.set_title("Canonical Space Map")
        self.ax_world.set_xlim(-0.5, 1.5)
        self.ax_world.set_ylim(-0.5, 1.5)
        self.ax_world.set_aspect('equal')
        
        # Learning visualization
        self.ax_learn = self.fig.add_subplot(gs[0, 1])
        self.ax_learn.set_title("Pattern Learning Progress")
        
        # Combat visualization
        self.ax_combat = self.fig.add_subplot(gs[1, 0])
        self.ax_combat.set_title("Combat Performance")
        
        # Stats visualization
        self.ax_stats = self.fig.add_subplot(gs[1, 1])
        self.ax_stats.set_title("Learning Metrics")
        
        # Initial draw of static elements
        self._draw_nav_map()
        
        # Show the plot
        plt.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.001)  # Small pause to update the UI
        
    def _draw_nav_map(self):
        """Draw the navigation map edges"""
        for node, neighbors in self.kb.level.nav.items():
            x, y, _ = node
            for neighbor in neighbors:
                nx, ny, _ = neighbor
                self.ax_world.plot([x, nx], [y, ny], 'k-', alpha=0.5)
        
        # Add nodes
        nodes = list(self.kb.level.nav.keys())
        x = [n[0] for n in nodes]
        y = [n[1] for n in nodes]
        self.ax_world.scatter(x, y, color='blue', s=100, zorder=10)
        
        # Add labels
        for node in nodes:
            x, y, _ = node
            self.ax_world.text(x+0.05, y+0.05, f"({x},{y})", fontsize=8)
    
    def update(self):
        """Manual update function called from main loop"""
        now = time.time()
        
        # Only update at specified interval
        if now - self.last_update < self.update_interval:
            return
            
        self.last_update = now
        
        # Update time history
        self.history['time'].append(now)
        
        # Clear the axes for redrawing
        self.ax_world.clear()
        self.ax_learn.clear()
        self.ax_combat.clear()
        self.ax_stats.clear()
        
        # Redraw the nav map (static elements)
        self._draw_nav_map()
        
        # Update titles
        self.ax_world.set_title("Canonical Space Map")
        self.ax_learn.set_title("Pattern Learning Progress")
        self.ax_combat.set_title("Combat Performance")
        self.ax_stats.set_title("Learning Metrics")
        
        # Update dynamic content
        self._update_world_view()
        self._update_learning_curves()
        self._update_combat_performance()
        self._update_stats_text()
        
        # Refresh the figure
        plt.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.001)  # Small pause to update the UI
        
    def _update_world_view(self):
        """Update the world map with current agent and opponent positions"""
        # Draw agent
        x, y, _ = self.kb.self.pos
        self.ax_world.scatter([x], [y], color='green', s=150, zorder=20)
        
        # Draw direction indicator (yaw)
        yaw_rad = math.radians(self.kb.self.yaw)
        dx, dy = 0.2 * math.cos(yaw_rad), 0.2 * math.sin(yaw_rad)
        self.ax_world.arrow(x, y, dx, dy, head_width=0.05, head_length=0.1, 
                            fc='green', ec='green')
        
        # Draw opponents
        for op_id, op in self.kb.opp.items():
            if op.confidence > 0.1:  # Only show if somewhat confident
                ox, oy, _ = op.pos
                alpha = min(1.0, op.confidence)
                
                weapon_colors = {'shotgun': 'red', 'rocket': 'orange', 'AR': 'purple'}
                color = weapon_colors.get(op.weapon, 'gray')
                
                # Draw opponent
                self.ax_world.scatter([ox], [oy], color=color, alpha=alpha, 
                                     s=100, zorder=15)
                
                # Add label
                self.ax_world.text(ox+0.05, oy-0.1, f"{op.weapon}", 
                                   fontsize=8, alpha=alpha)
                
                # Draw line of sight if visible
                if self.per.los(op.pos):
                    self.ax_world.plot([x, ox], [y, oy], 'g:', alpha=0.3)
        
        # Visualize active TIP paths
        self._visualize_paths()
        
        # Set limits
        self.ax_world.set_xlim(-0.5, 1.5)
        self.ax_world.set_ylim(-0.5, 1.5)
        self.ax_world.set_aspect('equal')
        
    def _visualize_paths(self):
        """Visualize planned paths from active TIPs"""
        # Find all active TIPs with paths
        for thread in threading.enumerate():
            if isinstance(thread, CombatTIP) and hasattr(thread, 'current_path'):
                if thread.current_path and len(thread.current_path) > 1:
                    path = thread.current_path
                    
                    # Extract x, y coordinates for plotting
                    path_x = [p[0] for p in path]
                    path_y = [p[1] for p in path]
                    
                    # Get color based on pattern
                    pattern_key = thread.fp.key
                    weapon_colors = {'combat/shotgun': 'red', 'combat/rocket': 'orange'}
                    color = weapon_colors.get(pattern_key, 'blue')
                    
                    # Plot path
                    self.ax_world.plot(path_x, path_y, '--', color=color, 
                                       linewidth=1.5, alpha=0.6)
        
    def _update_learning_curves(self):
        """Update the learning curves plot"""
        self.ax_learn.set_xlabel("Encounters")
        self.ax_learn.set_ylabel("Value")
        
        # Update pattern history
        for k, fp in self.meta.patterns.items():
            if k not in self.history['patterns']:
                self.history['patterns'][k] = {
                    'seen': [fp.stats['seen']],
                    'aim_offset': [fp.aim_offset],
                    'flank_bias': [fp.flank_bias]
                }
            else:
                self.history['patterns'][k]['seen'].append(fp.stats['seen'])
                self.history['patterns'][k]['aim_offset'].append(fp.aim_offset)
                self.history['patterns'][k]['flank_bias'].append(fp.flank_bias)
        
        # Plot learning curves
        for k, data in self.history['patterns'].items():
            encounters = range(len(data['seen']))
            weapon_colors = {'combat/shotgun': 'red', 'combat/rocket': 'orange'}
            color = weapon_colors.get(k, 'blue')
            
            # Plot aim offset learning
            self.ax_learn.plot(encounters, data['aim_offset'], '-', 
                              color=color, label=f"{k} (aim)")
                              
            # Plot flank bias learning
            self.ax_learn.plot(encounters, data['flank_bias'], '--', 
                              color=color, alpha=0.7, label=f"{k} (flank)")
        
        # Add legend and grid
        if self.history['patterns']:
            self.ax_learn.legend(loc='upper left', fontsize='x-small')
            self.ax_learn.grid(True, linestyle='--', alpha=0.3)
    
    def _update_combat_performance(self):
        """Update the combat performance plot"""
        # Calculate accuracy by pattern
        shots_by_pattern = self.meta.metrics['shots_by_pattern']
        hits_by_pattern = self.meta.metrics['hits_by_pattern']
        
        if not shots_by_pattern:
            self.ax_combat.text(0.5, 0.5, "No combat data yet", 
                              ha='center', va='center', fontsize=12)
            return
            
        # Calculate current accuracy for each pattern
        patterns = []
        accuracies = []
        colors = []
        
        for pattern, shots in shots_by_pattern.items():
            if shots > 0:
                hits = hits_by_pattern.get(pattern, 0)
                accuracy = (hits / shots) * 100
                patterns.append(pattern.split('/')[-1])  # Just the weapon name
                accuracies.append(accuracy)
                
                weapon_colors = {'shotgun': 'red', 'rocket': 'orange'}
                colors.append(weapon_colors.get(patterns[-1], 'blue'))
        
        # Bar chart of accuracy by pattern
        if patterns:
            bars = self.ax_combat.bar(patterns, accuracies, color=colors, alpha=0.7)
            
            # Add value labels on top of bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                self.ax_combat.text(bar.get_x() + bar.get_width()/2., height + 2,
                                  f'{acc:.1f}%', ha='center', va='bottom')
            
            # Add reference line for baseline accuracy
            self.ax_combat.axhline(y=60, color='gray', linestyle='--', alpha=0.5,
                                 label="Initial accuracy (60%)")
            
            # Configure axes
            self.ax_combat.set_ylabel("Accuracy (%)")
            self.ax_combat.set_ylim(0, 100)
            self.ax_combat.legend()
            
            # Add total shots/hits info
            total_shots = self.meta.metrics['total_shots']
            total_hits = self.meta.metrics['hits']
            if total_shots > 0:
                overall_acc = (total_hits / total_shots) * 100
                self.ax_combat.set_title(f"Combat Performance: {overall_acc:.1f}% accuracy ({total_hits}/{total_shots})")
    
    def _update_stats_text(self):
        """Update the stats text display"""
        self.ax_stats.axis('off')
        
        # Create a text summary of the current state
        stats_text = "=== Fractal Pattern Learning Statistics ===\n\n"
        
        # Pattern stats
        for k, fp in self.meta.patterns.items():
            stats_text += f"{k:15} seen={fp.stats['seen']:3.0f}  aim_offset={fp.aim_offset:+.3f}  flank_bias={fp.flank_bias:.2f}\n"
        
        # Combat metrics
        total_shots = self.meta.metrics['total_shots']
        accuracy = 0
        if total_shots > 0:
            accuracy = (self.meta.metrics['hits'] / total_shots) * 100
            
        stats_text += f"\nCombat Metrics: "
        stats_text += f"Shots={total_shots} Hits={self.meta.metrics['hits']} "
        stats_text += f"Accuracy={accuracy:.1f}% Damage Taken={self.meta.metrics['damage_taken']}"
        
        # Active TIPs
        active_tips = sum(1 for t in threading.enumerate() if isinstance(t, CombatTIP))
        stats_text += f"\nActive Tasks: {active_tips} (+ {self.meta.queue.qsize()} queued)"
        
        # Show the statistics
        self.ax_stats.text(0.01, 0.99, stats_text, transform=self.ax_stats.transAxes,
                          verticalalignment='top', fontsize=10, family='monospace')
                          
    def close(self):
        """Close the visualization window"""
        plt.close(self.fig)

# ════════════════════════════════════════════════════════════════════════════
#  Wire-up and demo
# ════════════════════════════════════════════════════════════════════════════
if __name__=="__main__":
    # Set up a more interesting map for visualization
    class EnhancedDummyEngine(DummyEngine):
        def __init__(self):
            # Create a slightly more complex map
            a=(0,0,0); b=(1,0,0); c=(1,1,0); d=(0,1,0)
            e=(0.5,0.5,0); f=(0.5,0,0); g=(0,0.5,0); h=(1,0.5,0); i=(0.5,1,0)
            self.nav={
                a:[b,d,f,g], b:[a,c,f,h], c:[b,d,h,i], d:[a,c,g,i],
                e:[f,g,h,i], f:[a,b,e], g:[a,d,e], h:[b,c,e], i:[c,d,e]
            }
            self._tick=0
            self._pos = (0,0,0)
            self._yaw = 0
            
        def get_self(self):
            # Random movement for more interesting visualization
            nodes = list(self.nav.keys())
            if random.random() < 0.1:  # Occasionally change position
                self._pos = random.choice(nodes)
            # Slowly rotate
            self._yaw = (self._yaw + 5) % 360
            return dict(pos=self._pos, yaw=self._yaw, hp=100)
            
        def visible(self):
            # Generate one opponent every second tick, alternating weapons
            self._tick+=1
            if self._tick%2==0:
                wid="shotgun" if self._tick%4==0 else "rocket"
                # Random position for opponent
                pos = random.choice(list(self.nav.keys()))
                return [dict(type="opponent",id=f"op{self._tick}",pos=pos,
                             yaw=random.randint(0,359),weapon=wid)]
            return []
    
    eng=EnhancedDummyEngine()
    lvl=CanonicalLevel(eng.get_nav())
    kb =KnowledgeBase(lvl)
    per=Perception(eng,kb)
    act=Action(eng,kb)
    meta=CoreAgent(kb,per,act)
    
    # Initialize visualizer
    vis = FractalVisualizer(kb, meta)
    
    # Run simulation
    try:
        for tick in range(30):  # Run for 30 ticks
            per.tick()
            # Submit a TIP per visible opponent
            for op in list(kb.opp.values()):
                if op.confidence>0.5: 
                    meta.encounter(op)
            
            # Update visualization
            vis.update()
            
            # Progress indicator
            print(f"Tick {tick+1}/30 complete")
            time.sleep(0.3)  # Slightly longer to observe visualization
            
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    finally:
        # Show learned pattern data
        print("\n=== Learned Fractal Patterns ===")
        for k,fp in meta.patterns.items():
            print(f"{k:15} seen={fp.stats['seen']:3.0f}  aim_offset={fp.aim_offset:+.3f}  flank_bias={fp.flank_bias:.2f}")
        
        # Keep plot open until user closes
        plt.ioff()
        plt.show()