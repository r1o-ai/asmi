#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Shape {
    pub channels: usize,
    pub width: usize,
    pub height: usize,
    pub batch: usize,
}

impl Shape {
    pub fn channels(channels: usize) -> Self {
        Self { channels, width: 1, height: 1, batch: 1 }
    }

    pub fn spatial(channels: usize, height: usize, width: usize) -> Self {
        Self { channels, width, height, batch: 1 }
    }

    pub fn total_elements(&self) -> usize {
        self.channels * self.width * self.height * self.batch
    }
}
