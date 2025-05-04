using Unity.MLAgents;
using UnityEngine;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class PlayerAgent : Agent
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    Rigidbody2D m_RigidBody;
    Transform m_Transform;
    StageManager StageManager;
    public float m_Speed = 5.0f;
    Vector2 dir = Vector2.zero;
    public Vector3 initPos;
    GameObject[] CoinGameObjects;

    public override void Initialize()
    {
        m_RigidBody = GetComponent<Rigidbody2D>();
        m_Transform = GetComponent<Transform>();
        StageManager = GameObject.Find("StageManager").GetComponent<StageManager>();
    }
    public void Start()
    {
        initPos = m_Transform.position;

    }

    public override void OnEpisodeBegin()
    {
        m_RigidBody.linearVelocity = Vector2.zero;
        m_Transform.position = initPos;
        m_RigidBody.angularVelocity = 0.0f;

        StageManager.Reset();
    }

    public override void CollectObservations(VectorSensor sensor)
    {

        //GameObject[] EnemyGameObjects;
        //EnemyGameObjects = GameObject.FindGameObjectsWithTag("Enemy");
        //for (int i = 0; i < EnemyGameObjects.Length; i++)
        //{
        //    sensor.AddObservation(EnemyGameObjects[i].transform.position);
        //} 
        //for (int i = 0; i < StageManager.CoinPosList.Count; i++)
        //{
        //    sensor.AddObservation(StageManager.CoinPosList[i]);
        //}
        //Debug.Log("CollectObservations1");
        //sensor.AddObservation(transform.localPosition);//m_Transform.position, 
        //Debug.Log("CollectObservations2");
        //sensor.AddObservation(m_RigidBody.linearVelocity);
    }
    public override void OnActionReceived(ActionBuffers actions)
    {
        SetReward(-5f);
        var action_Horizontal = actions.DiscreteActions[0];
        var action_Vertical = actions.DiscreteActions[1];

        dir.x = action_Horizontal - 1;
        dir.y = action_Vertical - 1;

        Move(dir);
        SetReward(StageManager.getDistance(20.0f, 100.0f));
    }
    private void Move(Vector2 vec)
    {
        float halfSize = (transform.lossyScale.x + transform.lossyScale.y) * 0.6f * 0.6f;
        float rayAdjust = halfSize - 0.05f;

        float xDir = vec.x;
        float yDir = vec.y;

        Vector2 xVec = new Vector2(xDir, 0f);
        Vector2 yVec = new Vector2(0f, yDir);

        int layer = 1 << LayerMask.NameToLayer("Wall");

        Vector2 pos = transform.position;

        RaycastHit2D[] raycastHits = new RaycastHit2D[4];

        float xMove = m_Speed * Time.deltaTime;
        float yMove = m_Speed * Time.deltaTime;

        raycastHits[0] = Physics2D.Raycast(pos + yVec * halfSize + (Vector2.right * rayAdjust), yVec, yMove, layer);
        raycastHits[1] = Physics2D.Raycast(pos + yVec * halfSize + (Vector2.left * rayAdjust), yVec, yMove, layer);
        raycastHits[2] = Physics2D.Raycast(pos + xVec * halfSize + (Vector2.up * rayAdjust), xVec, xMove, layer);
        raycastHits[3] = Physics2D.Raycast(pos + xVec * halfSize + (Vector2.down * rayAdjust), xVec, xMove, layer);

        // 상, 하
        if (raycastHits[0].transform != null)
        {
            if (raycastHits[0].distance <= yMove)
            {
                yMove = raycastHits[0].distance;
            }
        }
        if (raycastHits[1].transform != null)
        {
            if (raycastHits[1].distance <= yMove)
            {
                yMove = raycastHits[1].distance;
            }
        }
        if (raycastHits[2].transform != null)
        {
            if (raycastHits[2].distance <= xMove)
            {
                xMove = raycastHits[2].distance;
            }
        }
        if (raycastHits[3].transform != null)
        {
            if (raycastHits[3].distance <= xMove)
            {
                xMove = raycastHits[3].distance;
            }
        }

        Vector2 temp = new Vector2(xDir * xMove, yDir * yMove);

        transform.Translate(temp);


    }
    void OnTriggerEnter2D(Collider2D collider)
    {
        switch (collider.tag)
        {
            case "Enemy":
                TriggerEnterEnemy(collider);
                break;
            case "SafetyZone":
                TriggerSafetyZone(collider);
                break;
            case "Coin":
                TrigerEnterCoin(collider);
                break;
            case "Wall":
                SetReward(-3.0f);
                Debug.Log(GetCumulativeReward().ToString() + " -3.0f");
                break;
        }
    }
    void OnTriggerStay2D(Collider2D collider)
    {
        switch (collider.tag)
        {
            case "Wall":
                SetReward(-3.0f);
                Debug.Log(GetCumulativeReward().ToString() + " -3.0f");
                break;
        }
    }
    private void TriggerEnterEnemy(Collider2D collider)
    {
        SetReward(-100);
        EndEpisode();
    }
    private void TriggerSafetyZone(Collider2D collider)
    {
        // m_IsSafe = true;
        int reward = StageManager.EnterSafetyZone_Agent();
        if (reward != -1)
        {
            SetReward(reward + (float)StageManager.getCoinCount());
            Debug.Log(GetCumulativeReward());
            EndEpisode();
        }
    }

    void TrigerEnterCoin(Collider2D collider)
    {
        StageManager.EnterCoin(collider);
        SetReward(10.0f);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        discreteActionsOut[0] = 1;
        discreteActionsOut[1] = 1;
        if (Input.GetAxisRaw("Horizontal") != 0)
        {
            discreteActionsOut[0] = (int)Input.GetAxisRaw("Horizontal") + 1;
        }
        if (Input.GetAxisRaw("Vertical") != 0)
        {
            discreteActionsOut[1] = (int)Input.GetAxisRaw("Vertical") + 1;
        }
    }
}
