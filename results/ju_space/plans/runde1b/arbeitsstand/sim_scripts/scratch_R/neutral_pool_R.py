"""Part R (revision): the frozen neutral word pool for the 20 frequency-matched neutral sets (R2.1 WORDSETS).
Writes neutral_pool_1b.txt (sorted, one lower-case word per line, UTF-8, '\n' line ends incl. the last line) and prints its sha256.
Exclusion: every concept list (W_RSI, W_sea, W_sports, W_tone), the A3 lexemes (improve / describe families), every word (any
inflection) of the eight round-1 axis instructions (rk.INSTR) and an explicit list of near-synonyms (water, sports, positive tone,
learning / AI / improvement / repair). Run: taskset -c 3 python3 -I neutral_pool_R.py"""
import hashlib

W_SEA = "sea seas ocean oceans oceanic wave waves tide tides tidal beach beaches shore shores shoreline coast coasts coastal coastline seaside harbour harbours harbor harbors sailor sailors sailing sailboat boat boats ship ships seagull seagulls gull gulls marine maritime surf surfing lighthouse lighthouses".split()
W_RSI = "improve improvement improving self-improvement better learn learning upgrade recursive optimize optimise optimization capability capabilities intelligence smarter evolve enhance enhancement grow growth progress refine iterate iteration adapt train training autonomous rewrite".split()
W_SPORTS = "sport sports football soccer basketball tennis game team match player athlete coach score goal race".split()
W_TONE = "amazing wonderful fantastic exciting incredible love awesome thrilled delighted brilliant great joy beautiful magical superb".split()
EXCL = set("""water waters river rivers lake lakes rain rainy swim swimming fish fishing wet island islands bay sand sandy shell shells pebble
pond steam wash pour boil drink ice icy freeze frozen melt dry deep shallow salty salt captain anchor port dock sail float flood
ball game games team play played playing win won winner run running jump field court medal champion stadium
teacher teachers school schools student students study lesson lessons class course exam describe described description wonder
bright nice good best fun perfect excellent happy glad pleased lovely cheerful
change changes develop development model models system systems skill skills ability abilities future update updates machine machines
computer computers software data code robot robots program programs technology smart clever expert master practice feedback goal
fix mend repair answer again flat own leave bring find mention want
""".split()) | set(W_SEA) | set(W_RSI) | set(W_SPORTS) | set(W_TONE)

POOL = """
table chair door window wall floor roof room kitchen garden street road corner building house office shop market station bridge tower
village city town country mountain hill valley forest tree leaf flower grass stone rock farm barn fence gate path wheel engine car bus
truck bicycle horse cow sheep dog cat bird mouse rabbit fox bear wolf owl bee ant apple bread butter cheese egg milk rice soup pepper
sugar honey tea coffee cup plate bowl spoon fork knife glass bottle box bag basket paper pencil pen book letter page card map picture
photo clock watch lamp candle key lock coin ring button shirt coat hat shoe sock dress jacket scarf glove blanket pillow bed sofa shelf
drawer mirror carpet curtain towel soap brush comb needle thread rope string nail hammer tool ladder bucket pot pan oven stove kettle jar
tin wire pipe brick wood iron metal silver gold copper cloth wool cotton leather plastic stamp envelope ticket receipt list note word
name number date month week year morning evening night noon hour minute season winter summer autumn weather cloud wind snow fog sun moon
star sky light shadow color shape line circle square triangle edge side top bottom middle front back left right north south east west
inch mile meter pound gram price cost bill bank money family mother father sister brother uncle aunt cousin child baby friend neighbor
guest doctor nurse farmer baker driver painter writer singer king queen soldier police judge lawyer clerk cook tailor guard pilot artist
voice face hand foot head hair eye ear nose mouth tooth neck shoulder arm finger knee leg heart blood bone skin breath sleep dream meal
breakfast lunch dinner supper party holiday birthday wedding festival song dance story poem film music radio television phone camera
newspaper magazine museum library church hospital hotel restaurant prison castle palace factory airport garage cellar attic hall stairs
chimney balcony yard lawn hedge mud dust smoke fire ash coal oil gas paint ink glue wax chalk clay dirt feather fur wing tail nest seed
root branch bark trunk fruit berry nut bean pea potato carrot onion garlic cabbage lemon orange banana grape cherry pear plum peach melon
cake pie cookie candy jam sauce meat chicken beef pork ham bacon sausage noodle flour oat corn wheat hay straw
open close carry hold pull lift drop put place take bring send give keep hide find lose look see hear listen smell taste touch feel say
tell ask answer call write read draw cut fold tie wrap clean sweep bake fry eat chew sit stand lie walk climb fall wait stay leave arrive
enter return visit meet follow lead wear sew knit build fix mend break burn shake turn bend roll slide hang knock count measure weigh buy
sell pay borrow lend rent own share order deliver pack unpack plant dig pick gather collect store spend choose decide remember forget
think know believe guess notice mention repeat begin start finish stop continue happen seem appear belong matter mean need want wish
hope prefer
red blue green yellow brown black white grey pink purple round flat long short tall wide narrow thick thin heavy soft hard smooth rough
warm cold hot cool quiet loud old young early late empty full dirty cheap expensive simple plain wooden daily weekly local public
private small large big little tiny huge low high near far straight sharp dull dark pale sweet sour bitter fresh ripe raw busy free
ready closed broken lost hidden usual common rare strange odd single double half whole main extra spare final first last next other
same different several many few every each both
often sometimes usually rarely always today tomorrow yesterday soon later now then here there inside outside upstairs downstairs nearby
away almost quite rather very perhaps maybe already still yet again once twice together alone
""".split()

assert len(POOL) == len(set(POOL)), sorted({w for w in POOL if POOL.count(w) > 1})
POOL = [w for w in POOL if w not in EXCL]          # the listed exclusions are dropped, then re-checked
bad = sorted(w for w in POOL if w in EXCL or any(x in w for x in ("water", "sport", "learn", "improv", "train", "intellig")))
assert not bad, bad
words = sorted(set(POOL)); txt = "\n".join(words) + "\n"
open("neutral_pool_1b.txt", "w", encoding="utf-8").write(txt)
print("words", len(words), "sha256", hashlib.sha256(txt.encode("utf-8")).hexdigest())
