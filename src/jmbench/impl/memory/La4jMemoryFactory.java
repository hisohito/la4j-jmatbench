/*
 * Copyright (c) 2009-2011, Peter Abeles. All Rights Reserved.
 *
 * This file is part of JMatrixBenchmark.
 *
 * JMatrixBenchmark is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3
 * of the License, or (at your option) any later version.
 *
 * JMatrixBenchmark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with JMatrixBenchmark.  If not, see <http://www.gnu.org/licenses/>.
 */

package jmbench.impl.memory;

import jmbench.impl.wrapper.La4jBenchmarkMatrix;
import jmbench.interfaces.BenchmarkMatrix;
import jmbench.interfaces.MemoryFactory;
import jmbench.interfaces.MemoryProcessorInterface;

import org.la4j.linear.LinearSystem;
import org.la4j.matrix.Matrices;
import org.la4j.matrix.Matrix;
import org.la4j.matrix.dense.Basic1DMatrix;
import org.la4j.vector.Vector;

/**
 * @author Peter Abeles
 * @author Julia Kostyukova
 */
public class La4jMemoryFactory implements MemoryFactory {

	private static final long serialVersionUID = 1L;

	@Override
	public MemoryProcessorInterface svd() {
		return new SVD();
	}

	public static class SVD implements MemoryProcessorInterface {
		@Override
		public void process(BenchmarkMatrix[] inputs,
				BenchmarkMatrix[] outputs, long numTrials) {
			Matrix matA = inputs[0].getOriginal();

			for (long i = 0; i < numTrials; i++) {
				matA.decompose(Matrices.SINGULAR_VALUE_DECOMPOSITOR);
			}
		}
	}

	@Override
	public MemoryProcessorInterface eig() {
		return new Eig();
	}

	public static class Eig implements MemoryProcessorInterface {
		@Override
		public void process(BenchmarkMatrix[] inputs,
				BenchmarkMatrix[] outputs, long numTrials) {
			Matrix matA = inputs[0].getOriginal();

			for (long i = 0; i < numTrials; i++) {
				matA.decompose(Matrices.EIGEN_DECOMPOSITOR);
			}
		}
	}

	@Override
	public MemoryProcessorInterface invertSymmPosDef() {
		return new InvSymmPosDef();
	}

	public static class InvSymmPosDef implements MemoryProcessorInterface {
		@Override
		public void process(BenchmarkMatrix[] inputs,
				BenchmarkMatrix[] outputs, long numTrials) {
			Matrix matA = inputs[0].getOriginal();
			for (long i = 0; i < numTrials; i++) {
				matA.inverse(Matrices.DEFAULT_INVERTOR);
			}
		}
	}

	@Override
	public MemoryProcessorInterface add() {
		return new Add();
	}

	public static class Add implements MemoryProcessorInterface {
		@Override
		public void process(BenchmarkMatrix[] inputs,
				BenchmarkMatrix[] outputs, long numTrials) {
			Matrix matA = inputs[0].getOriginal();
			Matrix matB = inputs[1].getOriginal();

			for (long i = 0; i < numTrials; i++) {
				matA.unsafe_add(matB);
			}
		}
	}

	@Override
	public MemoryProcessorInterface mult() {
		return new Mult();
	}

	public static class Mult implements MemoryProcessorInterface {
		@Override
		public void process(BenchmarkMatrix[] inputs,
				BenchmarkMatrix[] outputs, long numTrials) {
			Matrix matA = inputs[0].getOriginal();
			Matrix matB = inputs[1].getOriginal();

			for (long i = 0; i < numTrials; i++) {
				matA.unsafe_multiply(matB);
			}
		}
	}

	@Override
	public MemoryProcessorInterface multTransB() {
		return new MulTranB();
	}

	public static class MulTranB implements MemoryProcessorInterface {
		@Override
		public void process(BenchmarkMatrix[] inputs,
				BenchmarkMatrix[] outputs, long numTrials) {
			Matrix matA = inputs[0].getOriginal();
			Matrix matB = inputs[1].getOriginal();

			for (long i = 0; i < numTrials; i++) {
				matA.unsafe_multiply(matB.transpose());
			}
		}
	}

	@Override
	public MemoryProcessorInterface solveEq() {
		return new Solve();
	}

	@Override
	public MemoryProcessorInterface solveLS() {
		return new Solve();
	}

	public static class Solve implements MemoryProcessorInterface {
		@Override
		public void process(BenchmarkMatrix[] inputs,
				BenchmarkMatrix[] outputs, long numTrials) {
			Matrix matA = inputs[0].getOriginal();
			Matrix matB = inputs[1].getOriginal();

			Vector vecB = La4jBenchmarkMatrix.toVector(matB);
			for (long i = 0; i < numTrials; i++) {
				LinearSystem system = Matrices.asLinearSystem(matA, vecB);
				system.solve(Matrices.DEFAULT_SOLVER);
			}
		}
	}

	@Override
	public void configure() {
	}

	@Override
	public BenchmarkMatrix create(int numRows, int numCols) {
		return new La4jBenchmarkMatrix(new Basic1DMatrix(numRows, numCols));
	}

	@Override
	public BenchmarkMatrix wrap(Object matrix) {
		return new La4jBenchmarkMatrix((Matrix) matrix);
	}
}
