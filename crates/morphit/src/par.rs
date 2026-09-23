//! Data-parallel loops. With the `parallel` feature these are rayon's; without
//! it the same method names run sequentially on the calling thread (the wasm
//! build). Every call site is an ordered map/collect or an independent
//! for_each, so both give identical results.

#[cfg(feature = "parallel")]
pub use rayon::prelude::*;

#[cfg(not(feature = "parallel"))]
pub use seq::*;

#[cfg(not(feature = "parallel"))]
mod seq {
    /// Sequential stand-in for rayon's slice methods.
    pub trait ParallelSlice<T> {
        fn par_iter(&self) -> std::slice::Iter<'_, T>;
        fn par_iter_mut(&mut self) -> std::slice::IterMut<'_, T>;
        fn par_chunks_mut(&mut self, chunk_size: usize) -> std::slice::ChunksMut<'_, T>;
    }

    impl<T> ParallelSlice<T> for [T] {
        fn par_iter(&self) -> std::slice::Iter<'_, T> {
            self.iter()
        }
        fn par_iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
            self.iter_mut()
        }
        fn par_chunks_mut(&mut self, chunk_size: usize) -> std::slice::ChunksMut<'_, T> {
            self.chunks_mut(chunk_size)
        }
    }

    /// Sequential stand-in for rayon's `into_par_iter` on index ranges.
    pub trait IntoParallelIterator {
        type Iter: Iterator;
        fn into_par_iter(self) -> Self::Iter;
    }

    impl IntoParallelIterator for std::ops::Range<usize> {
        type Iter = std::ops::Range<usize>;
        fn into_par_iter(self) -> Self::Iter {
            self
        }
    }
}
