//! Parquet sink + rejected log — mirror of src/sink/writer.py. TDD §4.6.

use crate::pipeline::StageResult;

pub struct Sink;

impl Sink {
    pub fn write_kept(&mut self, _doc_id: &str, _text: &str, _quality: f32) -> anyhow::Result<()> {
        todo!("TDD §4.6")
    }

    pub fn write_rejected(&mut self, _result: &StageResult) -> anyhow::Result<()> {
        todo!("TDD §4.6")
    }
}
