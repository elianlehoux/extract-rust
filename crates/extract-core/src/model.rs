//! Language-model trait used by the annotation pipeline.

use crate::types::ScoredOutput;
use crate::Error;

/// Backend that turns prompts into scored text outputs.
pub trait LanguageModel {
    fn infer(&mut self, batch_prompts: &[String]) -> Result<Vec<Vec<ScoredOutput>>, Error>;

    fn model_id(&self) -> Option<&str> {
        None
    }
}

/// Scripted backend for tests: each `infer` call consumes the next response.
#[derive(Debug, Clone)]
pub struct ScriptedModel {
    script: Vec<Vec<Vec<ScoredOutput>>>,
    cursor: usize,
    pub calls: Vec<Vec<String>>,
}

impl ScriptedModel {
    pub fn new(script: Vec<Vec<Vec<ScoredOutput>>>) -> Self {
        Self {
            script,
            cursor: 0,
            calls: Vec::new(),
        }
    }

    pub fn from_outputs(outputs: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let script = outputs
            .into_iter()
            .map(|o| vec![vec![ScoredOutput::new(o)]])
            .collect();
        Self::new(script)
    }
}

impl LanguageModel for ScriptedModel {
    fn infer(&mut self, batch_prompts: &[String]) -> Result<Vec<Vec<ScoredOutput>>, Error> {
        self.calls.push(batch_prompts.to_vec());
        if self.cursor >= self.script.len() {
            return Err(Error::InferenceOutput(
                "No scored outputs from language model.".into(),
            ));
        }
        let out = self.script[self.cursor].clone();
        self.cursor += 1;
        Ok(out)
    }
}

/// Model that maps each prompt to an output string.
pub struct MappingModel<F> {
    map: F,
    pub calls: Vec<Vec<String>>,
}

impl<F> MappingModel<F>
where
    F: FnMut(&str) -> String,
{
    pub fn new(map: F) -> Self {
        Self {
            map,
            calls: Vec::new(),
        }
    }
}

impl<F> LanguageModel for MappingModel<F>
where
    F: FnMut(&str) -> String,
{
    fn infer(&mut self, batch_prompts: &[String]) -> Result<Vec<Vec<ScoredOutput>>, Error> {
        self.calls.push(batch_prompts.to_vec());
        Ok(batch_prompts
            .iter()
            .map(|p| vec![ScoredOutput::new((self.map)(p))])
            .collect())
    }
}
