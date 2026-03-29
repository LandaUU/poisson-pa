import std.stdio;
import std.math : log, pow, exp;
import std.random : Xorshift, unpredictableSeed, uniform01;
import std.algorithm : min;
import std.array : array;
import std.conv : to;

struct GraphParams
{
	double c;
	double p;
	int deltaIn;
	int deltaOut;
}

struct Edge
{
	int from;
	int to;
}

// --- Lambda functions (match Python) ---
double lambdaF(int k, double c)
{
	// Python: np.power(k, c) * np.log(8 * k)
	return pow(cast(double) k, c) * log(8.0 * k);
}

double lambdaConst(int k, double c)
{
	// Python: return 10
	return 10.0;
}

final class Graph
{
private:
	GraphParams params;
	Xorshift rng;

	double function(int k, double c) lambdaFunc;

	// Node state
	bool[] hasMessage; // per node
	int[] inDeg; // per node
	int[] outDeg; // per node

	int round_ = 0;

public:
	// Per-round increments
	int[] N; // new nodes per round
	int[] S; // newly informed nodes per round

	// Cumulative totals (same semantics as Python: running total of N and S)
	int[] NTotal;
	int[] STotal;

	this(double c, double p, int deltaIn, int deltaOut,
		double function(int, double) lambdaFunc_ = &lambdaF,
		ulong seed = 0)
	{
		params = GraphParams(c, p, deltaIn, deltaOut);
		lambdaFunc = lambdaFunc_;
		rng = Xorshift(cast(int) seed);

		if (seed == 0)
		{
			rng.seed(unpredictableSeed);
		}
	}

	Graph initializeDefault()
	{
		// Matches init_graph(): nodes 0 and 1 with self-loops; node 0 has message.
		// :contentReference[oaicite:2]{index=2}
		clearState();

		addNode(true); // node 0
		addNode(false); // node 1

		// self-loops: 0->0 and 1->1
		applyEdges([Edge(0, 0), Edge(1, 1)]);
		// initial_N=2, initial_S=1 in Python
		N ~= 2;
		S ~= 1;
		NTotal ~= 2;
		STotal ~= 1;

		round_ = 0;
		return this;
	}

	// If you later need a custom initializer analogous to passing a NetworkX graph,
	// you can add an initializeFromEdges(...) that takes initial node count + edges + message flags.

	Graph evolute(int rounds, int kTrunc = 100_000)
	{
		foreach (step; 1 .. rounds + 1)
		{
			const currentRound = round_ + step;

			const lambdaVal = lambdaFunc(currentRound, params.c);

			// Python samples M via truncated pmf weights; equivalent is Poisson(lambda)+1.
			// We use inversion sampler for Poisson; ok for typical lambda sizes in your runs.
			int dM = poisson(lambdaVal) + 1;
			// writeln("Delta M_", currentRound, " = ", dM); // optional debug

			const n0 = nodeCount(); // probabilities fixed at start of the round (as in Python)
			auto pIn = inProbabilities(n0);
			auto pOut = outProbabilities(n0);

			int newNodes = 0;
			Edge[] edges;
			edges.reserve(dM);

			// Build edges
			foreach (_; 0 .. dM)
			{
				if (uniform01(rng) < params.p)
				{
					// alpha-case: add new node, connect new -> existing (no loops)
					int newId = addNode(false);
					int v = sampleCategorical(pIn); // target existing
					edges ~= Edge(newId, v);
					newNodes += 1;
				}
				else
				{
					// beta-case: edge between existing nodes
					int w = sampleCategorical(pOut);
					int v = sampleCategorical(pIn);
					edges ~= Edge(w, v);
				}
			}

			// Apply edges to degrees
			applyEdges(edges);

			// Message spread: if edge (w -> v) and v has message while w not, then w becomes informed
			int newlyInformed = 0;
			foreach (e; edges)
			{
				if (hasMessage[e.to] && !hasMessage[e.from])
				{
					hasMessage[e.from] = true;
					newlyInformed += 1;
				}
			}

			N ~= newNodes;
			S ~= newlyInformed;

			// totals: same recurrence as Python: last + increment
			NTotal ~= (NTotal.length == 0 ? newNodes : NTotal[$ - 1] + newNodes);
			STotal ~= (STotal.length == 0 ? newlyInformed : STotal[$ - 1] + newlyInformed);
		}

		round_ += rounds;
		return this;
	}

	int nodeCount() const
	{
		return cast(int) hasMessage.length;
	}

	// --- Output helpers replacing draw_ns_statistic ---
	void writeStatisticCSV(string path) const
	{
		// columns: k, N_total, S_total, S_total/N_total
		auto f = File(path, "w");
		scope (exit)
			f.close();
		f.writeln("k,N_total,S_total,ratio");

		int len = cast(int) min(NTotal.length, STotal.length);
		foreach (i; 0 .. len)
		{
			double ratio = (NTotal[i] == 0) ? double.nan
				: cast(double) STotal[i] / cast(double) NTotal[i];
			f.writeln(i, ",", NTotal[i], ",", STotal[i], ",", ratio);
		}
	}

private:
	void clearState()
	{
		hasMessage.length = 0;
		inDeg.length = 0;
		outDeg.length = 0;

		N.length = 0;
		S.length = 0;
		NTotal.length = 0;
		STotal.length = 0;
	}

	int addNode(bool message)
	{
		int id = nodeCount();
		hasMessage ~= message;
		inDeg ~= 0;
		outDeg ~= 0;
		return id;
	}

	void applyEdges(scope const Edge[] edges)
	{
		foreach (e; edges)
		{
			// ensure arrays large enough (defensive, though addNode already extends)
			if (e.from >= nodeCount() || e.to >= nodeCount())
			{
				throw new Exception("Edge references non-existent node id.");
			}
			outDeg[e.from] += 1;
			inDeg[e.to] += 1;
		}
	}

	double[] inProbabilities(int n0)
	{
		double[] w;
		w.length = n0;
		double sum = 0.0;
		foreach (i; 0 .. n0)
		{
			w[i] = cast(double)(inDeg[i] + params.deltaIn);
			sum += w[i];
		}
		if (sum <= 0)
			throw new Exception("Non-positive normalization in inProbabilities.");
		foreach (i; 0 .. n0)
			w[i] /= sum;
		return w;
	}

	double[] outProbabilities(int n0)
	{
		double[] w;
		w.length = n0;
		double sum = 0.0;
		foreach (i; 0 .. n0)
		{
			w[i] = cast(double)(outDeg[i] + params.deltaOut);
			sum += w[i];
		}
		if (sum <= 0)
			throw new Exception("Non-positive normalization in outProbabilities.");
		foreach (i; 0 .. n0)
			w[i] /= sum;
		return w;
	}

	int sampleCategorical(scope const double[] probs)
	{
		// probs assumed normalized to sum ~ 1
		double u = uniform01(rng);
		double c = 0.0;
		foreach (i, p; probs)
		{
			c += p;
			if (u <= c)
				return cast(int) i;
		}
		return cast(int)(probs.length - 1); // numerical fallback
	}

	int poisson(double lambda)
	{
		// Inversion by cumulative summation:
		// P(K=k) = e^-λ λ^k / k!
		// Complexity ~ O(K) ~ λ, usually fine for your parameter ranges.
		if (lambda < 0)
			throw new Exception("Poisson lambda must be non-negative.");
		if (lambda == 0)
			return 0;

		double u = uniform01(rng);
		double p = exp(-lambda);
		double s = p;
		int k = 0;

		while (u > s)
		{
			k += 1;
			p *= lambda / k;
			s += p;
			// defensive break for extreme cases (should not trigger in normal regimes)
			if (k > 10_000_000)
				throw new Exception("Poisson sampler diverged; lambda too large?");
		}
		return k;
	}
}

void main()
{
	enum int ROUNDS = 20;

	double c = 0.4;
	double p = 0.5;

	int[] N_TOTAL;
	int[] S_TOTAL;
	bool first = true;

	foreach (i; 0 .. ROUNDS)
	{
		auto g = new Graph(c, p, 1, 1);
		g.initializeDefault();
		g.evolute(500);

		if (first)
		{
			N_TOTAL = g.NTotal.dup;
			S_TOTAL = g.STotal.dup;
			first = false;
		}
		else
		{
			int minLen = cast(int) min(N_TOTAL.length, g.NTotal.length);
			N_TOTAL.length = minLen;
			S_TOTAL.length = minLen;
			foreach (j; 0 .. minLen)
			{
				N_TOTAL[j] += g.NTotal[j];
				S_TOTAL[j] += g.STotal[j];
			}
		}
	}

	// Average (integer rounding down; if you need float averages, store double[] instead)
	foreach (j; 0 .. N_TOTAL.length)
	{
		N_TOTAL[j] /= ROUNDS;
		S_TOTAL[j] /= ROUNDS;
	}

	// Write CSV for plotting (replacement for draw_ns_statistic)
	{
		auto f = File("stat.csv", "w");
		scope (exit)
			f.close();
		f.writeln("k,N_total_avg,S_total_avg,ratio");
		foreach (k; 0 .. N_TOTAL.length)
		{
			double ratio = (N_TOTAL[k] == 0) ? double.nan
				: cast(double) S_TOTAL[k] / cast(double) N_TOTAL[k];
			f.writeln(k, ",", N_TOTAL[k], ",", S_TOTAL[k], ",", ratio);
		}
	}

	writeln("Done. Wrote stat.csv");
}
